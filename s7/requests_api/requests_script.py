import requests
# response = requests.get('https://api.github.com')
response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
)

response_img = requests.get('https://imgs.xkcd.com/comics/making_progress.png')


# if response.status_code == 200:
#     print('Success!')
# elif response.status_code == 404:
#     print('Not Found.')

# print(type(response.content))
# # print(response.json())
# print(response_img.content)

pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)
print(response.json())