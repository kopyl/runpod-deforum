import runpod
import requests
import time
import os


headers = {"Prefer": "respond-async", "Content-Type": "application/json"}


def check_api_availability_and_run_generation(payload):
    while True:
        try:
            response = requests.post(
                "http://127.0.0.1:5000/predictions",
                json=payload,
                headers=headers,
                allow_redirects=False,
            )
            if response.status_code == 202:
                break
            else:
                print("Waiting for API to become available...")
                time.sleep(0.2)
        except requests.exceptions.RequestException as e:
            print(f"API is not available, retrying in 200ms... ({e})")
            time.sleep(0.2)
        except Exception as e:
            return {"error": f"DEFORUM_ERROR: {e}"}


print("run handler")


def handler(event):

    _input = event.get("input")
    if _input is None:
        return {
            "error": "INPUT_NOT_PROVIDED",
        }

    s3_config = event.get("s3Config")
    if s3_config is None:
        return {
            "error": "S3_CONFIG_NOT_PROVIDED",
        }

    if (aws_access_key_id := s3_config.get("accessId")) is None:
        return {"error": "AWS_ACCESS_KEY_ID_NOT_PROVIDED"}

    if (aws_secret_access_key := s3_config.get("accessSecret")) is None:
        return {
            "error": "AWS_SECRET_ACCESS_KEY_NOT_PROVIDED",
        }

    if (aws_bucket_name := s3_config.get("bucketName")) is None:
        return {
            "error": "BUCKET_NOT_PROVIDED",
        }

    endpoint_url = s3_config.get("endpointUrl")

    _input.update(
        {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_bucket_name": aws_bucket_name,
            "endpoint_url": endpoint_url,
        }
    )

    if (webhook := event.get("webhook")) is None:
        return {
            "error": "WEBHOOK_NOT_PROVIDED",
        }

    payload = {
        "webhook": webhook,
        "webhook_events_filter": ["start", "completed"],
        "input": _input,
    }

    check_api_availability_and_run_generation(payload)

    while True:
        if os.path.isfile("/tmp/finished"):
            break
        print("Waiting for output file to be created uploaded to S3...")
        time.sleep(1)

    os.remove("/tmp/finished")

    return {
        "version": "1.06.2023-00:23",
    }


runpod.serverless.start({"handler": handler})
