from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Animal
from filer.admin.imageadmin import ImageAdmin
from filer import settings as filer_settings
from filer.utils.loader import load_model


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    pass


@admin.register(Animal)
class AnimalAdmin(admin.ModelAdmin):
    list_display = ["__str__", "id_code", "name", "cohort"]
    search_fields = ["id_code__istartswith"]


Image = load_model(filer_settings.FILER_IMAGE_MODEL)


class CustomImageAdmin(ImageAdmin):
    select_related = ["animal"]
    autocomplete_fields = ["animal"]


# Using build_fieldsets allows to easily integrate common field in the admin
# Don't define fieldsets in the ModelAdmin above and add the custom fields
# to the ``extra_main_fields`` or ``extra_fieldsets`` as shown below
CustomImageAdmin.fieldsets = CustomImageAdmin.build_fieldsets(
    extra_main_fields=("animal", "side"),
    extra_fieldsets=(
        # (
        #     "Subject Location",
        #     {
        #         "fields": ("subject_location",),
        #         "classes": ("collapse",),
        #     },
        # ),
    ),
)

# Unregister the default admin
admin.site.unregister(Image)
# Register your own
admin.site.register(Image, CustomImageAdmin)
