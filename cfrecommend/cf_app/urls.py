from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.login, name='login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('home/', views.home, name='home'),
    path('keloladata/', views.keloladata, name='keloladata'),
    path('tambah-pelanggan/', views.tambah_pelanggan, name='tambah_pelanggan'),
    path('kelolaitem/', views.kelolaitem, name='kelolaitem'),
    path('tambah-item/', views.tambah_item, name='tambah_item'),
    path('tambah-penilaian/', views.tambah_penilaian, name='tambah_penilaian'),
    path('hapus-item/<int:item_id>/', views.hapus_item, name='hapus_item'),
    path('edit-item/<int:item_id>/', views.edit_item, name='edit_item'),
    path('get-ratings-for-pelanggan/', views.get_ratings_for_pelanggan, name='get_ratings_for_pelanggan'),
]