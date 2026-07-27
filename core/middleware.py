from collections.abc import Callable

from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from filer.models import Folder


class RedirectToUserFolderMiddleware:
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # request.user is set by AuthenticationMiddleware (runs before this
        # middleware, see webibex/settings.py MIDDLEWARE) but isn't declared
        # on the base HttpRequest without django-stubs (pre-existing repo-wide
        # stub gap -- see docs/security-remediation-plan.md).
        # Check if the user is authenticated and not a superuser, and is
        # trying to access the Filer folder changelist page.
        if (
            request.user.is_authenticated  # pyright: ignore[reportAttributeAccessIssue]
            and not request.user.is_superuser  # pyright: ignore[reportAttributeAccessIssue]
            and request.path == reverse("admin:filer_folder_changelist")
        ):
            # Redirect to the user's specific folder
            user = request.user  # pyright: ignore[reportAttributeAccessIssue]
            main_folder_name = f"{user.username}_files"
            try:
                main_user_folder = get_object_or_404(
                    Folder, name=main_folder_name, owner=user
                )
                url = reverse(
                    "admin:filer-directory_listing",
                    kwargs={"folder_id": main_user_folder.id},
                )
                return redirect(url)
            except Http404:
                # If the folder doesn't exist, proceed normally or handle it as you wish
                pass

        # Continue processing the request
        response = self.get_response(request)
        return response
