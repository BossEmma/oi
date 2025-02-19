from django import forms
from django.contrib.auth.forms import UserCreationForm , SetPasswordForm
from django.contrib.auth.models import User
from django.forms import ModelForm , TextInput,EmailInput , PasswordInput,NumberInput
from django.contrib.auth import get_user_model
from phonenumber_field.formfields import PhoneNumberField
from phonenumber_field.widgets import PhoneNumberPrefixWidget
from django.core.exceptions import ValidationError
from .models import  UserInfo,Deposit,Withdraw,Photo


class RegisterUserForm(UserCreationForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class':'form-control',"placeholder":"name@example.com"}))
    #firstname= forms.CharField(max_length=50,widget=forms.TextInput(attrs={"class":"form-control","placeholder":"Enter First Name"}))
    #lastname= forms.CharField(max_length=50,widget=forms.TextInput(attrs={"class":"form-control","placeholder":"Enter Last Name"}))
    class Meta:
        model = User
        fields = ('username','email','password1','password2')
    def clean_email(self):
        emails=self.cleaned_data["email"]
        if User.objects.filter(email=emails).exists():
            raise ValidationError("Email Already Exist. Try Again")
        return emails
    def __init__(self,*arg,**kwarg):
        super(RegisterUserForm,self).__init__(*arg,**kwarg)
        self.fields['username'].widget.attrs['class']='form-control'
        self.fields['username'].widget.attrs['placeholder']='Enter username'
        self.fields['email'].widget.attrs['class']='form-control'
        self.fields['email'].widget.attrs['placeholder']='name@example.com'
        self.fields['password1'].widget.attrs['class']='form-control'
        self.fields['password1'].widget.attrs['placeholder']='Password '
        self.fields['password2'].widget.attrs['class']='form-control'
        self.fields['password2'].widget.attrs['placeholder']='Confirm Password'
        
        for fieldname in ['username']:
            self.fields[fieldname].help_text = ""
        for fieldname in ['password1']:
            self.fields[fieldname].help_text = ""

class PasswordForm(SetPasswordForm):
    class meta:
        model = get_user_model()
        fields = ['password1','password2']
    def __init__(self,*arg,**kwarg):
        super(PasswordForm,self).__init__(*arg,**kwarg)
        self.fields['new_password1'].widget.attrs['class']='input-box'
        self.fields['new_password2'].widget.attrs['class']='input-box'
        self.fields['new_password1'].widget.attrs['placeholder']='New Passworld'
        self.fields['new_password2'].widget.attrs['placeholder']='Confirm Password'

#
class UserForm(ModelForm):
    class Meta:
        model =    UserInfo
        fields = ("firstname","lastname",
                  "contact","nationality","address","currency","birth",)
    def __init__(self,*arg,**kwarg):
        super(UserForm,self).__init__(*arg,**kwarg)
        
        self.fields["firstname"].widget.attrs['placeholder']='Firstname'
        self.fields["firstname"].widget.attrs['class']='form-control'
        self.fields["lastname"].widget.attrs['placeholder']='Lastname'
        self.fields["lastname"].widget.attrs['class']='form-control'
        
        
        
      
        self.fields['contact'].widget=PhoneNumberPrefixWidget(
                                                country_attrs={'class':'select mb-2'},
                                                initial="US",
                                                country_choices=None,
                                                number_attrs={'class':'menu form-control',"placeholder":"(555) 555-1234"},
                                                )
        
        self.fields["nationality"].widget.attrs['placeholder']='Nationality'
        self.fields["nationality"].widget.attrs['class']='form-control'
        self.fields["address"].widget.attrs['placeholder']='Address'
        self.fields["address"].widget.attrs['class']='form-control'
        self.fields['birth'].widget = forms.DateInput(
            attrs={
                'class': 'datepicker form-control',  # Add a CSS class for styling (optional)
                'placeholder': '1900-01-31',
                'type': 'date',# Add a placeholder text (optional)
                # Add any other attributes you want to customize
            }
        )
        self.fields["currency"].widget.attrs['placeholder']='Currency'
        self.fields["currency"].widget.attrs['class']='form-control'

        
class DepositForm(ModelForm):
    class Meta:
        model =    Deposit
        fields = ("ammount","fund","pay_in",)
    def __init__(self,*arg,**kwarg):
        super(DepositForm,self).__init__(*arg,**kwarg)
        self.fields["ammount"].widget.attrs['placeholder']='USD 0.00'
        self.fields["ammount"].widget.attrs['class']='form-control input input-box'
        self.fields["fund"].widget.attrs['placeholder']='Choose Wallet To Fund'
        self.fields["fund"].widget.attrs['class']='form-control input input-box'
        self.fields["pay_in"].widget.attrs['placeholder']=''
        self.fields["pay_in"].widget.attrs['class']='form-control input input-box'
        
class WithdrawForm(ModelForm):
    class Meta:
        model =   Withdraw  
        fields = ("ammount","fromacc","wallet","address","account_name","account_number","paypal","cashapp","bank_name",)
    def __init__(self,*arg,**kwarg):
        super(WithdrawForm,self).__init__(*arg,**kwarg)
        self.fields["ammount"].widget.attrs['placeholder']='USD 0.00'
        self.fields["ammount"].widget.attrs['class']='form-control input input-box'
        self.fields["fromacc"].widget.attrs['placeholder']=''
        self.fields["wallet"].widget.attrs['class']='form-control input input-box'
        self.fields["wallet"].widget.attrs['placeholder']=''
        self.fields["fromacc"].widget.attrs['class']='form-control input input-box'
        self.fields["address"].widget.attrs['placeholder']='Wallet Address'
        self.fields["address"].widget.attrs['class']='form-control input input-box'
        self.fields["account_name"].widget.attrs['class']='form-control input input-box'
        self.fields["account_name"].widget.attrs['placeholder']='Account Name'
        self.fields["account_number"].widget.attrs['class']='form-control input input-box'
        self.fields["account_number"].widget.attrs['placeholder']='Account Number'
        self.fields["bank_name"].widget.attrs['class']='form-control input input-box'
        self.fields["bank_name"].widget.attrs['placeholder']='Bank Name'
        self.fields["paypal"].widget.attrs['class']='form-control input input-box'
        self.fields["paypal"].widget.attrs['placeholder']='email@paypal.com'
        self.fields["cashapp"].widget.attrs['class']='form-control input input-box'
        self.fields["cashapp"].widget.attrs['placeholder']='Cashapp tag'

class PhotoForm(ModelForm):
    class Meta:
        model =    Photo
        fields = ("image",)
    def __init__(self,*arg,**kwarg):
        super(PhotoForm,self).__init__(*arg,**kwarg)
        self.fields["image"].widget.attrs['placeholder']='Profile Upload'
        self.fields["image"].widget.attrs['class']='form-control input input-box'
        
        
