from abc import abstractmethod, ABCMeta

from owlapy.model import OWLObject


class OWLObjectRenderer(metaclass=ABCMeta):
    """Abstract class with a render method to render an OWL Object into a string"""
    @abstractmethod
    def set_short_form_provider(self, short_form_provider) -> None:
        """Configure a short form provider that shortens the OWL objects during rendering

        Args:
            short_form_provider: short form provider
        """
        pass

    @abstractmethod
    def render(self, o: OWLObject) -> str:
        """Render OWL Object to string

        Args:
            o: OWL Object

        Returns:
            String rendition of OWL object
        """
        pass
