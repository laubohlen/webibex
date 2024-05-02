from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.translation import gettext_lazy as _
from .models import User, Animal
from filer.admin.imageadmin import ImageAdmin
from filer.admin.folderadmin import FolderAdmin
from filer import settings as filer_settings
from filer.models import Folder
from filer.utils.loader import load_model


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    pass


@admin.register(Animal)
class AnimalAdmin(admin.ModelAdmin):
    list_display = ["__str__", "id_code", "name", "cohort"]
    search_fields = ["id_code__istartswith"]


Image = load_model(filer_settings.FILER_IMAGE_MODEL)
# Folder = load_model(filer_settings.FILER_FOLDER_MODEL)


class CustomFolderAdmin(FolderAdmin):
    actions = [
        "delete_files_or_folders",
        "move_files_and_folders",
        "copy_files_and_folders",
        "resize_images",
        "rename_files",
        "tag_left",
        "tag_right",
        "tag_other",
    ]

    def tag_left(self, request, files_queryset, folders_queryset):
        """
        Action which updates 'side' tag of an IbexImage.
        Manually updating each image object instead of using update() on the files_queryset,
        becuase the files_queryset works general for all files and has no access to 'side'
        of individual ibex images.
        """
        for f in files_queryset:
            f.side = "L"
            f.save()
        self.message_user(
            request, f"{len(files_queryset)} images we're successfully taged 'left'"
        )
        return None

    def tag_right(self, request, files_queryset, folders_queryset):
        """
        Action which updates 'side' tag of an IbexImage.
        Manually updating each image object instead of using update() on the files_queryset,
        becuase the files_queryset works general for all files and has no access to 'side'
        of individual ibex images.
        """
        for f in files_queryset:
            f.side = "R"
            f.save()
        self.message_user(
            request, f"{len(files_queryset)} images we're successfully taged 'right'"
        )
        return None

    def tag_other(self, request, files_queryset, folders_queryset):
        """
        Action which updates 'side' tag of an IbexImage.
        Manually updating each image object instead of using update() on the files_queryset,
        becuase the files_queryset works general for all files and has no access to 'side'
        of individual ibex images.
        """
        for f in files_queryset:
            f.side = "O"
            f.save()
        self.message_user(
            request, f"{len(files_queryset)} images we're successfully taged 'other'"
        )
        return None

    pass


admin.site.unregister(Folder)
admin.site.register(Folder, CustomFolderAdmin)


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
