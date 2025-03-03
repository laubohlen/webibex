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
    Returns a URL for a page with a next parameter set to the current URL.
    Using the next parameter allows to redirect to the original URL after a task has been completed,
    independent of where the task has been originated from.
    """
    url = reverse(viewname, args=args, kwargs=kwargs)
    request = context["request"]
    next_param = request.get_full_path()
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'next': next_param})}"
