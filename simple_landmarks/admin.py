from typing import ClassVar

from django.contrib import admin

from .models import Landmark


@admin.register(Landmark)
class LandmarkAdmin(admin.ModelAdmin):
    search_fields: ClassVar[list[str]] = ["landmark"]
