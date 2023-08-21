import pytest
from django.contrib.auth import get_user_model


@pytest.mark.django_db
def test_create_user():
    User = get_user_model()
    user = User.objects.create_user(
        username="dinkythecat", email="dinky@thecat.com", password="testpass123"
    )
    assert user.username == "dinkythecat"
    assert user.email == "dinky@thecat.com"
    assert user.is_active == True
    assert user.is_staff == False
    assert user.is_superuser == False


@pytest.mark.django_db
def test_create_superuser():
    User = get_user_model()
    admin_user = User.objects.create_superuser(
        username="superadmin", email="superadmin@email.com", password="testpass123"
    )
    assert admin_user.username == "superadmin"
    assert admin_user.email == "superadmin@email.com"
    assert admin_user.is_active == True
    assert admin_user.is_staff == True
    assert admin_user.is_superuser == True
