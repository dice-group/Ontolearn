from abc import abstractmethod, ABCMeta

from ontolearn.owlapy.model import OWLObject


class OWLObjectRenderer(metaclass=ABCMeta):
    @abstractmethod
    def set_short_form_provider(self, short_form_provider) -> None:
        pass

    @abstractmethod
    def render(self, o: OWLObject) -> str:
        pass
