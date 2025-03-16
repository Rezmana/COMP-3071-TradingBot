import asyncio
from twikit import Client

USERNAME = '@rnmtesting872'
EMAIL = 'rnmtesting872@gmail.com'
PASSWORD = 'ShiddyTwitterDevOptions2#'

# Initialize client
client = Client('en-US')

async def main():
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD,
    )

asyncio.run(main())