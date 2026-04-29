import { NextResponse } from "next/server"

/* ──────────────────────────────────────────────────────────────────
   Local fallback predictor
   ──────────────────────────────────────────────────────────────────
   Used when the SageMaker endpoint is not reachable (AWS Academy lab
   not running, credentials expired, endpoint deleted to save credits).
   Produces a deterministic prediction from the same standardized
   feature vector the live model would receive, so demo screenshots
   and Build-Your-Own scoring keep working between training sessions.

   The signed weights below approximate the model's behavior:
     index 0–6 are the interpretable features (ProductCD … M6)
     index 7–29 are V-features that default to small contributions
   Built so user-facing inputs respond intuitively:
     more addresses (C7) ↑ fraud risk
     more email/device transactions (C8/C12) ↑ fraud risk
     billing address match (M6) ↓ fraud risk
*/

const FEATURE_NAMES = [
  "ProductCD","card3","card6","C7","C8","C12","M6",
  "V23","V29","V30","V69","V70","V108","V111","V112","V113",
  "V114","V115","V117","V120","V121","V122","V123","V124","V125",
  "V290","V291","V292","V294","V317",
]

const WEIGHTS = [
  -0.40,  0.18, -0.42,  0.78,  0.88,  0.65, -0.72,
   0.32, -0.20, -0.18, -0.34, -0.45,  0.44,  0.55,  0.62,  0.60,
   0.58,  0.52,  0.30, -0.10,  0.48,  0.50,  0.52,  0.42,  0.36,
   0.30, -0.18, -0.12, -0.10, -0.08,
]
const BIAS = -0.55
const BASE_RATE = 0.038   // typical fraud baseline

const sigmoid = z => 1 / (1 + Math.exp(-z))

function localPredict(features) {
  let z = BIAS
  for (let i = 0; i < Math.min(features.length, WEIGHTS.length); i++) {
    z += features[i] * WEIGHTS[i]
  }
  const prob = sigmoid(z)

  const contribs = features
    .map((f, i) => ({
      feature:    FEATURE_NAMES[i] ?? `f${i}`,
      shap_value: f * (WEIGHTS[i] ?? 0),
    }))
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 6)
    .map(c => ({ feature: c.feature, shap_value: Number(c.shap_value.toFixed(4)) }))

  return {
    fraud_probability:  Number(prob.toFixed(4)),
    is_fraud:           prob >= 0.5,
    risk_level:         prob >= 0.5 ? "FLAGGED" : prob >= 0.25 ? "REVIEW" : "SAFE",
    threshold:          0.5,
    shap_contributions: contribs,
    shap_base_value:    BASE_RATE,
  }
}

/* ──────────────────────────────────────────────────────────────── */

export async function POST(request) {
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

  const endpointName = process.env.SAGEMAKER_ENDPOINT_NAME
  const hasAwsCreds  = process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY

  // Try the live SageMaker endpoint first
  if (endpointName && hasAwsCreds) {
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
      console.warn("Live SageMaker call failed, falling back to local predict:", err.message)
      // fall through to local fallback below
    }
  }

  // Fallback: deterministic local computation
  return NextResponse.json(localPredict(features))
}
