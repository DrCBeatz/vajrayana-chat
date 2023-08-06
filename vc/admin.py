from django.contrib import admin
from .models import Model, Expert, Conversation, Message, Document


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0  # how many rows to show


class ConversationAdmin(admin.ModelAdmin):
    inlines = [
        MessageInline,
    ]


admin.site.register(Model)
admin.site.register(Expert)
admin.site.register(Conversation, ConversationAdmin)
admin.site.register(Message)
admin.site.register(Document)
