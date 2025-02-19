from ast import mod
from doctest import FAIL_FAST
from email.policy import default
from django.db import models
from phonenumber_field.modelfields import PhoneNumberField
from django.conf import settings
from django.core import validators
from django.core.exceptions import ValidationError
from django_countries.fields import CountryField
from django.contrib.auth import get_user_model
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from django.contrib.auth.models import User
import secrets
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from django.core.validators import MinValueValidator
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from datetime import date, timedelta

import random
import string
from .email_otp import text,otptext
Userer    = settings.AUTH_USER_MODEL
password = 'death2025'
sender_email = 'services@ultragreentrade.com'

MARKET_TYPE =(
    ("CRYPTO","CRYPTO"),("STOCKS","STOCKS"),("CURRENCIES","CURRENCIES"),("INDICES","INDICES"),
)

def validate_age(value):
    min_age = 7
    today = date.today()
    min_birth_date = today - timedelta(days=(min_age * 365.25))  # Account for leap years

    if value > today:
        raise ValidationError(
            _('%(value)s is a future date. Please enter a date in the past.'),
            params={'value': value},
        )
    elif value > min_birth_date:
        raise ValidationError(
            _('%(value)s is less than %(min_age)d years ago. You must be at least %(min_age)d years old.'),
            params={'value': value, 'min_age': min_age},
        )

class WalletBalance(models.Model):
    trading_balance = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2)
    bitcoin_mining = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=6)
    ethereum_mining = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=6)
    dogecoin_mining = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=6)
    binance_coin_mining = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=6)
    cosmos_mining = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=6)
    verified = models.BooleanField(default=False)
    user   = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    date = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=("-date",)

    def __str__(self) -> str:
        return f"User: {self.user}"
    
class CopyTrader(models.Model):
    name = models.CharField(max_length=200,null=False,blank=False)
    win_rate = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2)
    profit_share = models.PositiveIntegerField(blank=False,null=False)
    asset_under_management = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2)
    days = models.PositiveIntegerField(blank=False,null=False)
    profile_image = models.ImageField(upload_to='images/')
    date = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=("-date",)

    def __str__(self) -> str:
        return f"User: {self.name}"
    
@receiver(pre_delete, sender=CopyTrader)
def delete_image(sender, instance, **kwargs):
    # Delete the associated image file
    instance.profile_image.delete(False)
    
class CopiedTrade(models.Model):
    name = models.CharField(max_length=200,null=False,blank=False)
    win_rate = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2)
    profit_share = models.PositiveIntegerField(blank=False,null=False,default=40)
    profile_image = models.ImageField(upload_to='images/')
    date = models.DateTimeField(auto_now_add=True)
    user   = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    class Meta:
        ordering=("-date",)

    def __str__(self) -> str:
        return f"User: {self.user}"

class Coin(models.Model):
    name = models.CharField(max_length=200,null=False,blank=False)
    value = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=6)
    mine_rate = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2)
    image = models.ImageField(upload_to='images/')
    date = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=("-date",)

    def __str__(self) -> str:
        return f"User: {self.name}"
    
@receiver(pre_delete, sender=Coin)
def delete_coin(sender, instance, **kwargs):
    # Delete the associated image file
    instance.image.delete(False)
    
class Market(models.Model):
    name = models.CharField(max_length=200,null=False,blank=False)
    symbol = models.CharField(max_length=200,null=False,blank=False)
    form = models.CharField(max_length=200,null=False,blank=False,choices=MARKET_TYPE)
    value = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=6)
    image = models.ImageField(upload_to='images/')
    date = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=("-date",)

    def __str__(self) -> str:
        return f"User: {self.name}"
@receiver(pre_delete, sender=Market)
def delete_market(sender, instance, **kwargs):
    # Delete the associated image file
    instance.image.delete(False)

class WatchList(models.Model):
    name = models.CharField(max_length=200,null=False,blank=False)
    symbol = models.CharField(max_length=200,null=False,blank=False)
    value = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=6)
    form = models.CharField(max_length=200,null=False,blank=False,choices=MARKET_TYPE)
    image = models.ImageField(upload_to='images/')
    date = models.DateTimeField(auto_now_add=True)
    user   = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    class Meta:
        ordering=("-date",)

    def __str__(self) -> str:
        return f"User: {self.name}"

CURRENCY=(
    ("USD","USD"),("EUR","EUR"),("CAD","CAD"),("AUD","AUD"),("GBP","GBP"),
    )  
class UserInfo(models.Model):
    firstname       = models.CharField(max_length=50 , blank=False)
    lastname       = models.CharField(max_length=50 , blank=False)
    birth = models.DateField(blank=False,null=False,validators=[validate_age])
    currency       = models.CharField(max_length=50 ,choices= CURRENCY, blank=False)
    email         = models.EmailField(blank=True,null=True ,default="test@gmail.com")
    contact       = PhoneNumberField(blank=False)
    nationality = CountryField(blank=False,null=False)
    address  = models.CharField(max_length=70, null=False,blank=False,help_text="describe your location")
    date_created = models.DateTimeField(auto_now_add=True)
    user          = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    class Meta:
        ordering=("-date_created",)
    def __str__(self):
        return f"{self.user}"
    
class Otp(models.Model):
    time = models.TimeField(auto_now_add=True)
    otp = models.CharField(max_length=100,blank=False,null=False)
    email = models.EmailField()
    verified = models.BooleanField(default=False)
    user   = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    class Meta:
        ordering=("-time",)

    def __str__(self) -> str:
        return f"otp: {self.time}"
    def save(self , *args,**kwargs) -> None:
        while not self.otp :
            otp = random.randrange(10001,50005,2)
            object_with_similar_ref  = Otp.objects.filter(otp=otp)
            if not object_with_similar_ref:
                self.otp = otp
                receiver_email =f"{self.email}"
                message = MIMEMultipart("alternative")
                message["Subject"] = "Account Activity"
                message["From"] = sender_email
                message["To"] = receiver_email
                try:
                    html = otptext.format(self.time,self.user,self.otp)
                    part1 = MIMEText(text, "plain")
                    part2 = MIMEText(html, "html")

                    message.attach(part1)
                    message.attach(part2)

                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL("mail.privateemail.com", 465, context=context) as server:
                        server.login(sender_email, password)
                        server.sendmail(
                            sender_email, receiver_email, message.as_string()
                        )
                    super(Otp, self).save(*args, **kwargs)
                except:
                    pass

DEPOSIT_TO =(
    ("TRADING BALANCE","TRADING BALANCE"),("BITCOIN MINIG","BITCOIN MINIG"),
    ("DOGECOIN MINING","DOGECOIN MINING"),("ETHEREUM MINING","ETHEREUM MINING"),
    ("BINANCE COIN MINING","BINANCE COIN MINING"),("COSMO (ATOM) MINING","COSMO (ATOM) MINING"),
)

def limit_value(value):
    val=int(value)
    if val >= 500:
        return value
    else:
        raise ValidationError("Deposit above 500(USD)")

PAYMETHOD =  (
    ("SOL Solana Solana","SOL Solana Solana"),("USDT Tether TRC20","USDT Tether TRC20"),
    ("LTC Litecoin","LTC Litecoin"),("SHIB Shiba ERC20","SHIB Shiba ERC20"),
    ("ETH Ethereum ERC20","ETH Ethereum ERC20"),("DOGE Dogecoin","DOGE Dogecoin"),
    ("BNB Binance coin Binance Smart Chain","BNB Binance coin Binance Smart Chain"),("BTC Bitcoin ","BTC Bitcoin "),
)  
class Deposit(models.Model):
    ammount = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2,validators=[MinValueValidator(500.00)])
    fund = models.CharField(max_length=100,blank=False,null=False,choices=DEPOSIT_TO,default=1) # type: ignore
    pay_in = models.CharField(max_length = 200,blank=False,null=False,choices=PAYMETHOD,default=1) # type: ignore
    status = models.BooleanField(default=False)
    date_created = models.DateTimeField(auto_now_add=True)
    user          = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    class Meta:
        ordering=("-date_created",)
    def __str__(self):
        return f"{self.user}"

class Withdraw(models.Model):
    ammount = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2,validators=[MinValueValidator(500.00)])
    fromacc = models.CharField(max_length=100,blank=False,null=False,choices=DEPOSIT_TO,default=1) # type: ignore
    wallet = models.CharField(max_length = 200,blank=True,null=True,choices=PAYMETHOD,default=1) # type: ignore
    account_name = models.CharField(max_length = 200,blank=True,null=True,default="None")
    account_number = models.CharField(max_length = 200,blank=True,null=True,default="None")
    address = models.CharField(max_length=500,blank=True,null=True)
    paypal = models.EmailField(blank=True,null=True,default="none@ultragreen.com")
    bank_name = models.CharField(max_length = 200,blank=True,null=True,default="None")
    cashapp = models.CharField(max_length = 200,blank=True,null=True,default="None")

    status = models.BooleanField(default=False)
    date_created = models.DateTimeField(auto_now_add=True)
    user          = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    class Meta:
        ordering=("-date_created",)
    def __str__(self):
        return f"{self.user}"
    def save(self, *args, **kwargs):
        # Check if the withdrawal amount is greater than the account balance
        balance = WalletBalance.objects.get(user=self.user)
        if self.ammount > balance.trading_balance:
            raise ValidationError("Withdrawal amount cannot be greater than account balance.")

        super().save(*args, **kwargs)
        
        # Decrease the user's balance
        balance.trading_balance -= self.ammount
        balance.save()
    
STATES =   (
    ("OPEN TRADE","OPEN TRADE"),("CLOSED TRADE","CLOSED TRADE"),
)  
LOSS_GAIN =   (
    ("LOSS TRADE","LOSS TRADE"),("GAIN TRADE","GAIN TRADE"),
)  
class OpenClosedTrade(models.Model):
    ammount = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2)
    up = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2)
    down = models.DecimalField(max_digits=999,blank=False,null=False,decimal_places=2)
    name = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    loss_or_gain = models.CharField(max_length=500,blank=False,null=False,default=1,choices=LOSS_GAIN)
    trade_type = models.CharField(max_length=500,blank=False,null=False,default=1,choices=STATES)
    pair = models.CharField(max_length=500,blank=False,null=False,default="BTCUSD")
    date_created = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=("-date_created",)
    def __str__(self):
        return f"{self.name}"
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        
        # Increase the user's balance
        balance, created = WalletBalance.objects.get_or_create(user=self.name)
        balance.trading_balance -= self.ammount
        balance.save()

class DepositCoin(models.Model):
    qrcode = models.ImageField(upload_to='images/')
    pair = models.CharField(max_length=500,blank=False,null=False,default="wallet Address")
    Coin = models.CharField(max_length = 200,blank=False,null=False,choices=PAYMETHOD,default=1) # type: ignore
    date_created = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=("-date_created",)
    def __str__(self):
        return f"{self.Coin}"
    
@receiver(pre_delete, sender=DepositCoin)
def delete_depcoin(sender, instance, **kwargs):
    # Delete the associated image file
    instance.qrcode.delete(False)

class Photo(models.Model):
    image = models.ImageField(upload_to='images/')
    user          = models.ForeignKey(Userer , default=1 , null=True, on_delete=models.SET_NULL)
    date_created = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering=("-date_created",)
    def __str__(self):
        return f"{str(self.user)}"
    
@receiver(pre_delete, sender=Photo)
def delete_ph(sender, instance, **kwargs):
    # Delete the associated image file
    instance.image.delete(False)
