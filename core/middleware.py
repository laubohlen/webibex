from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from filer.models import Folder


class RedirectToUserFolderMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check if the user is authenticated and not a superuser
        if request.user.is_authenticated and not request.user.is_superuser:
            # Check if the user is trying to access the Filer folder changelist page
            if request.path == reverse("admin:filer_folder_changelist"):
                # Redirect to the user's specific folder
                user = request.user
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
                except Folder.DoesNotExist:
                    # If the folder doesn't exist, proceed normally or handle it as you wish
                    pass

        # Continue processing the request
        response = self.get_response(request)
        return response
