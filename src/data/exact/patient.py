from .core import Core
from .resources import metadata


class ExactPatient:
    def __init__(self, specifier):
        self._specifier = specifier
        assert self._specifier in list(
            metadata().patient_specifier
        ), f"Patient {specifier} does not exist"
        self._metadata = None

    @property
    def core_metadata(self):
        if self._metadata is None:
            self._metadata = metadata().query("patient_specifier in @self._specifier")
        return self._metadata

    @property
    def core_specifiers(self):
        return list(self.core_metadata["core_specifier"])

    @property
    def num_benign_cores(self):
        return len(self.core_metadata.query('grade == "Benign"'))

    @property
    def all_cores_are_benign(self):
        return self.num_benign_cores == self.num_cores

    @property
    def num_cores(self):
        return len(self.core_metadata)

    def get_cores(self, as_list = False) -> list[Core]:
        if as_list:
            return [Core.create_core(specifier) for specifier in self.core_specifiers]
        return {specifier: Core.create_core(specifier) for specifier in self.core_specifiers}

    @staticmethod
    def sample_patient():
        """
        create sample patient object to speed up development
        """
        return ExactPatient("UVA-0007")

    