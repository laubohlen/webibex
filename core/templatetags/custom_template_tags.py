from django import template
from django.urls import reverse
from urllib.parse import urlencode

register = template.Library()


@register.filter
def dict_get(d, key):
    return d.get(key, "")


@register.simple_tag(takes_context=True)
def post_task_redirect(context, viewname, *args, **kwargs):
    """
    Returns a URL for a page with a next parameter. If a next parameter already
    exists in the current request, it is preserved to chain deeper redirects.
    """
    request = context["request"]
    # If the next parameter already exists, preserve it; otherwise, use current URL
    original_next = request.GET.get("next", request.get_full_path())
    url = reverse(viewname, args=args, kwargs=kwargs)
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'next': original_next})}"
