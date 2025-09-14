from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # This line tells Django to use the URL patterns from your 'redactor_app'.
    path('', include('redactor_app.urls')),
]
