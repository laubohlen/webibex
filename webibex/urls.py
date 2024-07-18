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

admin.site.site_header = "Webibex Admin"


urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("allauth.urls")),
    path("filer/", include("filer.urls")),
    path("", welcome_view, name="welcome"),
    path("upload/", upload_view, name="upload"),
    path("observed/", observed_animal_view, name="observed-animals"),
    path("unobserved/", unobserved_animal_view, name="unobserved-animals"),
    path("unidentified/", unidentified_images_view, name="unidentified-images"),
    path("to-landmark/", to_landmark_images_view, name="to-landmark"),
    re_path(
        r"^landmark_horn/(?P<oid>[0-9]+)/$", landmark_horn_view, name="landmark-horn"
    ),
    re_path(r"^landmark_eye/(?P<oid>[0-9]+)/$", landmark_eye_view, name="landmark-eye"),
    re_path(
        r"^finished_landmarks/(?P<oid>[0-9]+)/$",
        finished_landmark_view,
        name="finished-landmarks",
    ),
    re_path(r"^chip/(?P<oid>[0-9]+)/$", chip_view, name="chip"),
]

if settings.DEBUG:
    urlpatterns += [path("test/", test_view, name="test")]
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
