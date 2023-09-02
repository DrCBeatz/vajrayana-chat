from django.core.cache import cache
from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from .views import get_embeddings
from .models import Expert, Document
from django.db.models import Max



@receiver(user_logged_in)
def user_logged_in_callback(sender, request, user, **kwargs):
    experts = Expert.objects.all()

    last_modified_timestamps = {}
    for expert in experts:
        embeddings = get_embeddings(expert_name=expert.name)
        sanitized_name = expert.name.replace(
            " ", "_"
        )  # sanitize the name to ensure it's a valid cache key
        cache.set(f"embeddings_{expert.name}", embeddings, None)

        last_modified = Document.objects.filter(expert=expert).aggregate(
            Max("last_modified")
        )["last_modified__max"]
        if last_modified:
            last_modified_timestamps[expert.name] = last_modified

    cache.set("last_modified_timestamps", last_modified_timestamps, None)
