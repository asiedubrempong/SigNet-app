from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO 

from fastai import *
from fastai.vision import *

model_file_url = 'https://drive.google.com/uc?export=download&id=1WYa4JxG5UayTuHBJWRzJleOjr1ecIUn6'
model_file_name = 'model'

classes = ['Ahead only', 'Beware of ice/snow', 'Bicycles crossing', 'Bumpy road', 'Children crossing', 'Dangerous curve to the left', 
	'Dangerous curve to the right', 'Double curve', 'End of all speed and passing limits', 'End of no passing', 
	'End of no passing by vehicles over 3.5 metric tons', 'End of speed limit (80km/h)', 'General caution', 'Go straight or left', 
	'Go straight or right', 'Keep left', 'Keep right', 'No entry', 'No passing', 'No passing for vehicles over 3.5 metric tons',
 	'No vehicles', 'Pedestrians', 'Priority road', 'Right-of-way at the next intersection', 'Road narrows on the right',
	'Road work', 'Roundabout mandatory', 'Slippery road', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'Speed limit (20km/h)',
	'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
 	'Stop', 'Traffic signals', 'Turn left ahead', 'Turn right ahead', 'Vehicles over 3.5 metric tons prohibited',
	'Wild animals crossing', 'Yield']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)

