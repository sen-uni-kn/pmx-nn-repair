import os
from logging import info
from pathlib import Path
from typing import Optional, Tuple, Union, Final

import numpy as np
import pandas
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.integrate

from utils import sample_data

_DEFAULT_SEED: Final[int] = 187672899876819


class DatasetIDR(Dataset):
    """
    A dataset of samples of an indirect response (IDR) model.
    Each dataset is characterized by the R0, kout, Imax, IC50, WT, dose,
    Cl, V, and ka parameters of the model and the dataset size.

    Datasets are stored in csv files for later re-use.
    """

    def __init__(
        self,
        root: Optional[Union[str, os.PathLike]],
        size: int = 10000,
        samples_per_patient: int = 25,
        R0: float = 100.0,
        kout_distribution: str = "log-normal[0.5,0.3]",
        Imax: float = 1.0,
        IC50_distribution: str = "log-normal[2.5,0.3]",
        WT_distribution: str = "uniform[1.0,3.0]",
        dose_at_unit_WT: float = 50.0,
        Cl: float = 0.2,
        V: float = 2.0,
        ka: float = 0.5,
        t_max: float = 96.0,
        random_seed: int = _DEFAULT_SEED,
    ):
        """
        Load a IDR dataset from :code:`root`.
        If :code:`root` doesn't contain a dataset with the given parameters and size,
        generate such a dataset and store it in :code:`root`.

        Each dataset has four input attributes: dose, kout, IC50,
        and t (time) (in this order).
        It has one output :math:`R`.

        See :code:`utils.sample_data` for specifying the distributions.

        :param root: The directory where datasets are stored.
            If :code:`root` is None, the dataset isn't stored.
        :param size: The size of the dataset. The size needs to be a multiple
            of :code:`samples_per_patient`.
        :param samples_per_patient: How many time points are sampled for each patient.
            Each patient corresponds to one sampled combination of Cl, V, ka, and WT.
            Overall, the dataset contains :math:`\frac{size}{samples\_per\_patient}`
            patients.
        :param R0: The :math:`R_0` parameter of the IDR model.
        :param Imax: The :math:`I_{\max}` parameter of the IDR model.
        :param kout_distribution: The distribution of the kout parameter.
            For example "log-normal[0.5,0.3]" for a log-normal distribution with
            standard deviation :math:`0.3` and mean :math:`\ln(0.5)`,
            or "uniform[0.14,1.47]" for a uniform distribution with offset :math:`0.14`
            and spread :math:`1.47`.
            See :code:`utils.sample_data` for more details on specifying distributions.
        :param IC50_distribution: The distribution of the IC50 parameter,
            similarly to :code:`kout_distribution`.
        :param WT_distribution: The distribution of the WT parameter,
            similarly to :code:`kout_distribution`.
        :param dose_at_unit_WT: The dose at :math:`WT = 1.0`.
        :param Cl: The Cl parameter.
        :param V: The V parameter.
        :param ka: The ka parameter.
        :param t_max: The duration of the time window in which the IDR
            model is sampled.
            Each time point is sampled uniformly from :math:`[0.0, t\_max]`.
        :param random_seed: The random seed to use for generating the dataset.
        """
        if size % samples_per_patient != 0:
            raise ValueError(
                "Size needs to be a multiple of samples_per_patient. "
                f"Got size={size}, samples_per_patient={samples_per_patient}."
            )
        self.size = size
        self.samples_per_patient = samples_per_patient
        self.R0 = R0
        self.kout_distribution = kout_distribution
        self.Imax = Imax
        self.IC50_distribution = IC50_distribution
        self.WT_distribution = WT_distribution
        self.dose_at_unit_WT = dose_at_unit_WT
        self.Cl = Cl
        self.V = V
        self.ka = ka
        self.t_max = t_max
        self.seed = random_seed

        if root is None:
            self.dataset_path = None
            self.data_in, self.data_out = self._generate()
            self.data = torch.hstack([self.data_in, self.data_out])
            return

        dose_str = f"{dose_at_unit_WT}".replace(".", "_DOT_")
        file_name = (
            f"dataset_IDR_"
            f"N_{self.size}_"
            f"R0_{self.R0}_"
            f"kout_{self.kout_distribution}_"
            f"Imax_{self.Imax}_"
            f"IC50_{self.IC50_distribution}_"
            f"WT_{self.WT_distribution}_"
            f"dose_{dose_str}_"
            f"Cl_{self.Cl}_"
            f"V_{self.V}_"
            f"ka_{self.ka}_"
            f"tmax_{self.t_max}"
        )
        if self.seed != _DEFAULT_SEED:
            file_name += f"_seed_{self.seed}"
        file_name += ".csv"
        file_path = Path(root, file_name)
        if not file_path.exists():
            self.data_in, self.data_out = self._generate()
            file_path.parent.mkdir(exist_ok=True, parents=True)
            data = torch.hstack([self.data_in, self.data_out])
            data_df = pandas.DataFrame(data.numpy())
            data_df.columns = ("dose", "kout", "IC50", "t", "R")
            info(f"Storing dataset in {file_path}.")
            data_df.to_csv(file_path, index=False)
        else:
            info(f"Loading dataset from: {file_path}")
            data_df = pd.read_csv(file_path, index_col=False)
            data_in = data_df.iloc[:, :-1]
            data_out = data_df.iloc[:, -1]
            self.data_in = torch.as_tensor(data_in.to_numpy())
            self.data_out = torch.as_tensor(data_out.to_numpy()).reshape(-1, 1)
        self.dataset_path = file_path
        self.data_in = self.data_in.float()
        self.data_out = self.data_out.float()

    def _generate(self):
        info(
            f"Generating IDR dataset (size = {self.size}, R = {self.R0}, "
            f"Imax = {self.Imax}, kout: {self.kout_distribution}, "
            f"IC50: {self.IC50_distribution}, WT: {self.WT_distribution}, "
            f"dose at unit WT = {self.dose_at_unit_WT}, "
            f"t max = {self.t_max}, seed = {self.seed})"
        )
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        N = self.size // self.samples_per_patient
        M = self.samples_per_patient

        try:
            WT = sample_data(N, self.WT_distribution, rng)
        except ValueError as ex:
            raise ValueError(
                f"Failed sampling WT values from distribution {self.WT_distribution}"
            ) from ex
        dose = self.dose_at_unit_WT * WT

        try:
            kout = sample_data(N, self.kout_distribution, rng)
        except ValueError as ex:
            raise ValueError(
                f"Failed sampling kout values from distribution {self.kout_distribution}"
            ) from ex

        try:
            IC50 = sample_data(N, self.IC50_distribution, rng)
        except ValueError as ex:
            raise ValueError(
                f"Failed sampling IC50 values from distribution {self.IC50_distribution}"
            ) from ex

        # the last value is a placeholder for the time point t.
        data_in = torch.hstack([dose, kout, IC50, torch.zeros(N, 1)])
        data_in = data_in.repeat_interleave(M, dim=0)
        data_out = torch.empty(N * M, 1)
        # need to solve each ODE separately
        info("Computing outputs by simulating patients (solving ODE)...")
        for i in tqdm(range(N), desc="patients"):
            dose, kout, IC50, _ = data_in[i * M]
            ts = torch.rand(M, generator=rng) * self.t_max
            Rs = self.compute_output(dose, kout, IC50, ts)
            data_in[i*M:(i+1)*M, 3] = ts
            data_out[i*M:(i+1)*M, 0] = Rs

        info("Dataset generation finished.")
        return data_in, data_out

    def compute_output(
        self,
        dose: float,
        kout: float,
        IC50: float,
        ts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the output/concentration of a set of input values.
        The function is **not** vectorized, but can compute multiple
        time points (:code:`ts`) at once.

        Each invocation involves solving an ordinary differential equation.
        Therefore, it's highly recommended to use the capacity to sample
        multiple time-points at once.

        :param dose: The administered dose.
        :param kout: The kout value.
        :param IC50: The IC50 value.
        :param ts: The time points.
        :return: The output :math:`R` at times :code:`ts`.
        """
        # turn tensors into floats
        dose = float(dose)
        kout = float(kout)
        IC50 = float(IC50)
        ts_sorted, ts_indices = torch.sort(ts.squeeze(), dim=0)
        kin = self.R0 * kout

        def ode_fun(t, y):
            dy1 = -self.ka * y[0, :]
            dy2 = self.ka * y[0, :] - (self.Cl / self.V) * y[1, :]
            C = y[1, :] / self.V
            dy3 = kin * (1 - (self.Imax*C)/(IC50+C)) - kout * y[2, :]
            return np.array([dy1, dy2, dy3])

        # Using jac leads to some cryptic error in scipy.
        # def jac(t, y):
        #     C = y[1] / self.V
        #     y3dy2 = -kin * (
        #         self.Imax / self.V * (IC50 + y[1])
        #         - self.Imax * y[1] / np.square(self.V)
        #     ) / np.square(IC50 + C),
        #     return np.array([
        #         [-self.ka, 0.0, 0.0],
        #         [self.ka, -(self.Cl / self.V), 0.0],
        #         [0.0, y3dy2, -kout],
        #     ])

        y0 = np.array([dose, 0.0, self.R0])
        result = scipy.integrate.solve_ivp(
            ode_fun,
            t_span=(0, self.t_max),
            y0=y0,
            method="BDF",
            t_eval=ts_sorted.numpy(),
            vectorized=True,
            # jac=jac,
        )

        if result.status != 0:
            raise ValueError(f"Solving an ODE failed. {result}")
        Rs_sorted = result.y[2, :]

        Rs = torch.empty_like(ts)
        Rs[ts_indices] = torch.as_tensor(Rs_sorted, dtype=torch.float)
        return Rs

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_in[index], self.data_out[index]

    def __len__(self) -> int:
        return self.size
