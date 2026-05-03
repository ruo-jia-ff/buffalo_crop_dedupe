import requests
from requests.auth import HTTPBasicAuth
from time import sleep

from datetime import datetime, timedelta, timezone
import os

from azure.core.credentials import AzureNamedKeyCredential
from azure.data.tables import generate_account_sas as table_generate_account_sas, ResourceTypes as TB_ResourceTypes, AccountSasPermissions as TableSasPermissions

from dotenv import load_dotenv

# SAS TOKEN Generator class, it creates the token to access tables and file shares 
class SASTokenGenerator:
    def __init__(self, 
                 username='ML', 
                 password='CUDAisKING24',
                 api_resource='table',
                 api_address="http://100.64.0.11:8000/",
                 api_request="generate_sas_token?resource=",
                 write=True,
                 create=False,
                 delete=False
                ):
        
        self.username = username
        self.password = password
        self.resource = api_resource
        self.api_address = api_address
        self.api_request = api_request
        self.write = write
        self.create = create
        self.delete = delete

    def generate_sas_token_from_z11(self):
        
        # Construct the API request URL 
        api_url = f"{self.api_address}{self.api_request}{self.resource}"

        # Check for create and delete and write
        extras = [self.create, self.delete, self.write]
        hdr = ["&create=", "&delete=", "&write="]
        extras = [item[0] + str(item[1]).lower() for item in zip(hdr, extras)]
        extras = "".join(extras)
        
        api_url += extras

        try: 
            print(api_url)
            # Get the response and check if it is valid
            response = requests.get(api_url, auth=HTTPBasicAuth(self.username, self.password))
            response.raise_for_status()
            
            # Return SAS token
            return response.json()
        
        # Catch various errors
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # Log the HTTP error
        except requests.exceptions.RequestException as req_err:
            print(f"Error occurred during the request: {req_err}")  # Log other request errors
        except ValueError as json_err:
            print(f"JSON decode error: {json_err}")  # Log JSON decode errors
        except KeyError as key_err:
            print(f"Key error: {key_err}")  # Log missing key error
        except Exception as e:
            print(f"An unexpected error occurred generating the token: {e}")  # Log any other unexpected errors
    
    def generate_table_sas_token(self, 
                                 read=True, 
                                 write=True, 
                                 create=True, 
                                 delete=True,
                                 expiry_hours=48) -> dict:
        
        account_name = os.getenv("STORAGE_ACCOUNT")
        account_key = os.getenv("ACCESS_KEY")

        if not account_name or not account_key:
            raise ValueError("Please set STORAGE_ACCOUNT and ACCESS_KEY in your environment")

        credential = AzureNamedKeyCredential(account_name, account_key)

        start = datetime.now(timezone.utc) - timedelta(minutes=5)  # start 5 minutes ago
        expiry = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

        resources = TB_ResourceTypes(object=True, container=True)
        permissions = TableSasPermissions(read=read,
                                            add=write,
                                            update=write,
                                            create=create,
                                            delete=delete,
                                            list=True)  

        SAS_Token = table_generate_account_sas(credential=credential,
                                                resource_types=resources,
                                                permission=permissions,
                                                expiry=expiry,
                                                start = start)  

        return {'sas_token': SAS_Token, 'expiry': expiry}
    
    def generate_sas_token(self):

        token = self.generate_file_sas_token_backup()

        if token:
            return token 
        
        else:
            return self.generate_sas_token_from_z11()