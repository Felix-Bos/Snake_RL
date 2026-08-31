from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/checkpoints/', views.checkpoints_json, name='checkpoints_json'),
    path('api/obstacle-layouts/', views.list_obstacle_layouts_json, name='list_obstacle_layouts'),
    path('api/obstacle-layouts/save/', views.save_obstacle_layout, name='save_obstacle_layout'),
    path('api/obstacle-layouts/<str:name>/', views.load_obstacle_layout_json, name='load_obstacle_layout'),
]
