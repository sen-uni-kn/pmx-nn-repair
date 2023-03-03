import os
from logging import info
from pathlib import Path
from typing import Optional, Tuple, Union, Final

import pandas
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import sample_data

_DEFAULT_SEED: Final[int] = 437937598516331


class Dataset1CMPPO(Dataset):
    """
    A dataset of samples of the single compartment model
    for per-oral consumption (1-cmp PO model).
    Each dataset is characterized by the Cl, V, ka, WT, and dose parameters
    of the model and the dataset size.

    Datasets are stored in csv files for later re-use.
    """

    def __init__(
        self,
        root: Optional[Union[str, os.PathLike]],
        size: int = 10000,
        samples_per_patient: int = 25,
        Cl_distribution: str = "log-normal[0.2,0.3]",
        V_distribution: str = "log-normal[2.0,0.3]",
        ka_distribution: str = "log-normal[0.5,0.3]",
        WT_distribution: str = "uniform[1.0,3.0]",
        dose_at_unit_WT: float = 50.0,
        t_max: float = 24.0,
        random_seed: int = _DEFAULT_SEED,
    ):
        """
        Load a 1-cmp PO dataset from :code:`root`.
        If :code:`root` doesn't contain a dataset with the given parameters and size,
        generate such a dataset and store it in :code:`root`.

        Each dataset has five input attributes: dose, Cl, V, ka,
        and t (time) (in this order).
        It has one output: the concentration.

        See :code:`utils.sample_data` for specifying the distributions.

        :param root: The directory where datasets are stored.
            If :code:`root` is None, the dataset isn't stored.
        :param size: The size of the dataset. The size needs to be a multiple
            of :code:`samples_per_patient`.
        :param samples_per_patient: How many time points are sampled for each patient.
            Each patient corresponds to one sampled combination of Cl, V, ka, and WT.
            Overall, the dataset contains :math:`\frac{size}{samples\_per\_patient}`
            patients.
        :param Cl_distribution: The distribution of the CL parameter.
            For example "log-normal[0.2,0.3]" for a log-normal distribution with
            standard deviation :math:`0.3` and mean :math:`\ln(0.2)`,
            or "uniform[0.05,0.606]" for a uniform distribution with offset :math:`0.05`
            and spread :math:`0.606`.
            See :code:`utils.sample_data` for more details on specifying distributions.
        :param V_distribution: The distribution of the V parameter,
            similarly to :code:`CL_distribution`.
        :param ka_distribution: The distribution of the ka parameter,
            similarly to :code:`CL_distribution`.
        :param WT_distribution: The distribution of the WT parameter,
            similarly to :code:`CL_distribution`.
        :param dose_at_unit_WT: The dose at :math:`WT = 1.0`.
        :param t_max: The duration of the time window in which the 1 cmp PO
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
        self.Cl_distribution = Cl_distribution
        self.V_distribution = V_distribution
        self.ka_distribution = ka_distribution
        self.WT_distribution = WT_distribution
        self.dose_at_unit_WT = dose_at_unit_WT
        self.t_max = t_max
        self.seed = random_seed

        if root is None:
            self.dataset_path = None
            self.data_in, self.data_out = self._generate()
            self.data = torch.hstack([self.data_in, self.data_out])
            return

        dose_str = f"{dose_at_unit_WT}".replace(".", "_DOT_")
        file_name = (
            f"dataset_1_cmp_po_"
            f"N_{self.size}_"
            f"Cl_{self.Cl_distribution}_"
            f"V_{self.V_distribution}_"
            f"ka_{self.ka_distribution}_"
            f"WT_{self.WT_distribution}_"
            f"dose_{dose_str}_"
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
            data_df.columns = ("dose", "Cl", "V", "ka", "t", "concentration")
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
            f"Generating 1 CMP PO dataset (size = {self.size}, Cl: {self.Cl_distribution}, "
            f"V: {self.V_distribution}, ka: {self.ka_distribution}, "
            f"WT: {self.WT_distribution}, dose at unit WT = {self.dose_at_unit_WT}, "
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
            Cl = sample_data(N, self.Cl_distribution, rng)
        except ValueError as ex:
            raise ValueError(
                f"Failed sampling Cl values from distribution {self.Cl_distribution}"
            ) from ex
        try:
            V = sample_data(N, self.V_distribution, rng)
        except ValueError as ex:
            raise ValueError(
                f"Failed sampling V values from distribution {self.V_distribution}"
            ) from ex
        try:
            ka = sample_data(N, self.ka_distribution, rng)
        except ValueError as ex:
            raise ValueError(
                f"Failed sampling ka values from distribution {self.ka_distribution}"
            ) from ex

        data_in = torch.hstack([dose, Cl, V, ka])
        data_in = data_in.repeat_interleave(M, dim=0)
        t = torch.rand(N * M, 1, generator=rng) * self.t_max
        data_in = torch.hstack([data_in, t])

        info("Computing outputs...")
        dose, Cl, V, ka, t = data_in.T  # match dose, Cl, V, ka to t
        data_out = self.compute_output(dose, Cl, V, ka, t)
        info("Dataset generation finished.")
        return data_in, data_out.unsqueeze(1)

    @staticmethod
    def compute_output(
        dose: torch.Tensor,
        Cl: torch.Tensor,
        V: torch.Tensor,
        ka: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the output/concentration of a set of input values.
        The function is vectorized.
        Therefore, you can compute outputs in batches by supplying
        vectors for the arguments.

        :param dose: The administered dose.
        :param Cl: The Cl value.
        :param V: The V value.
        :param ka: The ka value.
        :param t: The time point.
        :return: The concentration at time :code:`t`.
        """
        kel = Cl / V
        return (
            (dose / V)
            * (ka / (ka - kel))
            * (torch.exp(-kel * t) - torch.exp(-ka * t))
        )

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data_in[index], self.data_out[index]

    def __len__(self) -> int:
        return self.size
