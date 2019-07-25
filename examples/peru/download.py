import wget

print('Downloading files ...')

url = 'https://sid.erda.dk/share_redirect/fcwjD77gUY/peru.npy'
wget.download(url, 'peru.npy')
print("")
url = 'https://sid.erda.dk/share_redirect/fcwjD77gUY/peru_ndmi_stack.grd'
wget.download(url, 'peru_ndmi_stack.grd')
print("")
