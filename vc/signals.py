from django.core.cache import cache
from django.contrib.auth.signals import user_logged_in
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .views import get_embeddings
from .models import Expert, Document
from django.db.models import Max


@receiver(post_save, sender=Document)
@receiver(post_delete, sender=Document)
def update_embeddings_cache(sender, instance, **kwargs):
    # Invalidate cache for specific expert
    expert_name = instance.expert.name
    cache_key = f"embeddings_{expert_name}"
    cache.delete(cache_key)

    last_modified_key = f"last_modified_{expert_name}"
    cache.delete(last_modified_key)


@receiver(user_logged_in)
def user_logged_in_callback(sender, request, user, **kwargs):
    experts = Expert.objects.all()

    last_modified_timestamps = {}
    for expert in experts:
        embeddings = get_embeddings(expert_name=expert.name)
        cache.set(f"embeddings_{expert.name}", embeddings, None)

        last_modified = Document.objects.filter(expert=expert).aggregate(
            Max("last_modified")
        )["last_modified__max"]
        if last_modified:
            last_modified_timestamps[expert.name] = last_modified

    cache.set("last_modified_timestamps", last_modified_timestamps, None)
