import { NextResponse } from "next/server"

export async function POST(request) {
  const endpointName = process.env.SAGEMAKER_ENDPOINT_NAME
  if (!endpointName) {
    return NextResponse.json(
      { error: "SAGEMAKER_ENDPOINT_NAME not configured" },
      { status: 503 }
    )
  }

  let features
  try {
    const body = await request.json()
    features = body.features
    if (!Array.isArray(features) || features.length !== 30) {
      return NextResponse.json(
        { error: "features must be an array of 30 numbers" },
        { status: 400 }
      )
    }
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 })
  }

  try {
    const { SageMakerRuntimeClient, InvokeEndpointCommand } =
      await import("@aws-sdk/client-sagemaker-runtime")

    const credentials = {
      accessKeyId:     process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    }
    if (process.env.AWS_SESSION_TOKEN) {
      credentials.sessionToken = process.env.AWS_SESSION_TOKEN
    }

    const client = new SageMakerRuntimeClient({
      region: process.env.AWS_REGION || "us-east-1",
      credentials,
    })

    const command = new InvokeEndpointCommand({
      EndpointName: endpointName,
      ContentType:  "application/json",
      Body:         Buffer.from(JSON.stringify({ features })),
    })

    const response = await client.send(command)
    const result   = JSON.parse(Buffer.from(response.Body).toString())
    return NextResponse.json(result)
  } catch (err) {
    console.error("SageMaker invoke error:", err)
    return NextResponse.json(
      { error: "Prediction failed", detail: err.message },
      { status: 502 }
    )
  }
}
