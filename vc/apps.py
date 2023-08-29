from django.apps import AppConfig


class VcConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "vc"

    def ready(self):
        import vc.signals
