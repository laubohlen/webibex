"""
URL configuration for webibex project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include, re_path
from core.views import *


admin.site.site_header = "Webibex"

urlpatterns = [
    path("webibex/", admin.site.urls),
    path("accounts/", include("allauth.urls")),
    path("filer/", include("filer.urls")),
    path("", welcome_view, name="welcome"),
    path("identification/", images_overview, name="images-overview"),
    path("upload/", image_upload, name="upload-images"),
    re_path(r"^image/(?P<oid>[0-9]+)/$", image_read, name="read-image"),
    re_path(r"^update-image/(?P<oid>[0-9]+)/$", image_update, name="update-image"),
    re_path(r"^update-delete/(?P<oid>[0-9]+)/$", image_delete, name="delete-image"),
    path("animals/", animals_overview, name="animals"),
    path("unidentified/", unidentified_images_view, name="unidentified-images"),
    path("results/", results_over_view, name="results-overview"),
    re_path(
        r"^result_default/(?P<oid>[0-9]+)/$",
        default_chip_compare_view,
        name="result-default",
    ),
    re_path(
        r"^result_refined/(?P<oid>[0-9]+)/$",
        project_chip_compare_view,
        name="result-refined",
    ),
    re_path(r"^run_again/(?P<oid>[0-9]+)/$", rerun_view, name="run-again"),
    re_path(r"^animal/(?P<oid>[0-9]+)/$", animal_images_view, name="animal"),
    re_path(
        r"^animal-own-images/(?P<oid>[0-9]+)/$",
        animal_images_owner_view,
        name="animal-own-images",
    ),
    re_path(r"^new-ibex/(?P<oid>[0-9]+)/$", created_animal_view, name="new-ibex"),
    path(
        "updated-ibex-images/",
        saved_animal_selection_view,
        name="saved-animal-selection",
    ),
    path("create-region/", create_region, name="create-region"),
    path("save-region/", save_region, name="save-region"),
    re_path(r"^region/(?P<oid>[0-9]+)/$", read_region, name="read-region"),
    re_path(r"^delete-region/(?P<oid>[0-9]+)/$", delete_region, name="delete-region"),
    re_path(r"^update-region/(?P<oid>[0-9]+)/$", update_region, name="update-region"),
    path("region-overview/", region_overview, name="region-overview"),
    re_path(
        r"^locate-image/(?P<oid>[0-9]+)/$",
        create_loaction,
        name="locate-image",
    ),
    path("multi-task/", multi_task_view, name="multi-task"),
    path("save-image-location/", save_image_location, name="save-image-location"),
    path("save-landmarks/", save_landmarks_view, name="save-landmarks"),
]

if settings.DEBUG:
    urlpatterns += [path("test/", test_view, name="test")]
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    try:    
        from debug_toolbar.toolbar import debug_toolbar_urls
        urlpatterns += debug_toolbar_urls()
    except ImportError:
        print("ImportError Debug toolbar")
    
