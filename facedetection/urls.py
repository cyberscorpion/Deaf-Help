

from django.contrib import admin
from django.urls import path
#from blog import views
#import face_detection_app2.views
from django.conf import settings 
from django.conf.urls.static import static

import face_detector
from face_detector import views
from face_detector.views import *

urlpatterns = [
    path('', views.home ,name="home"),
    path('face_detection/detect/', views.face_detection ,name="face_detect"),
    path('upload',upload_page),
    path('video',video_page),
    path('admin/', admin.site.urls),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
