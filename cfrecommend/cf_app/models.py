from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    no_telp = models.CharField(max_length=100)

class Item(models.Model):
    name = models.CharField(max_length=100)

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    item = models.ForeignKey(Item, on_delete=models.CASCADE)
    rating = models.FloatField(null=True, blank=True)