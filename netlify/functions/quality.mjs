/**
 * Netlify Function: /api/quality
 *
 * Proxies Claude vision requests for post-SHARP quality assessment.
 * Receives the original photo + a viewer screenshot (canvas capture)
 * and returns a structured quality report.
 *
 * POST body (JSON):
 *   originalBase64  – base64 JPEG of the original photo
 *   previewBase64   – base64 JPEG screenshot of the 3D viewer
 *   sceneData       – parsed scene analysis JSON (from /api/analyze)
 *   apiKey          – user's Anthropic key (ignored if env var is set)
 *   model           – Claude model slug
 */

const SYSTEM = `You are a 3D reconstruction quality reviewer. You will receive two images:
1. The original photograph used as input.
2. A screenshot of the 3D Gaussian Splatting scene generated from that photo.

Evaluate the quality and return ONLY valid JSON with this exact structure:
{
  "quality_score": <integer 1-10>,
  "quality_summary": "one sentence summary of the quality",
  "visible_artifacts": ["artifact1", "artifact2"],
  "inspection_checklist": ["item1", "item2", "item3"]
}
Be concise and honest.`;

export const handler = async (event) => {
  if (event.httpMethod === "OPTIONS") {
    return cors(200, "");
  }
  if (event.httpMethod !== "POST") {
    return cors(405, JSON.stringify({ error: "Method not allowed" }));
  }

  let body;
  try {
    body = JSON.parse(event.body || "{}");
  } catch {
    return cors(400, JSON.stringify({ error: "Invalid JSON body" }));
  }

  const {
    originalBase64,
    previewBase64,
    sceneData,
    apiKey,
    model = "claude-sonnet-4-5",
  } = body;

  const key = process.env.ANTHROPIC_API_KEY || apiKey;
  if (!key) return cors(400, JSON.stringify({ error: "No Anthropic API key provided." }));
  if (!originalBase64 || !previewBase64) {
    return cors(400, JSON.stringify({ error: "originalBase64 and previewBase64 are required." }));
  }

  const sceneContext = sceneData
    ? `Scene context: ${sceneData.scene_summary || ""}. Depth confidence: ${sceneData.depth_confidence || "unknown"}.`
    : "";

  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        max_tokens: 1024,
        system: SYSTEM,
        messages: [{
          role: "user",
          content: [
            {
              type: "image",
              source: { type: "base64", media_type: "image/jpeg", data: originalBase64 },
            },
            {
              type: "image",
              source: { type: "base64", media_type: "image/jpeg", data: previewBase64 },
            },
            {
              type: "text",
              text: `The first image is the original photo. The second is a screenshot of the 3D Gaussian Splatting result. ${sceneContext} Evaluate the 3D reconstruction quality and return only the JSON.`,
            },
          ],
        }],
      }),
    });

    if (!res.ok) {
      const err = await res.text().catch(() => "");
      return cors(res.status, JSON.stringify({ error: `Anthropic error ${res.status}`, detail: err.slice(0, 300) }));
    }

    const data    = await res.json();
    const rawText = data.content?.[0]?.text || "{}";
    const jStart  = rawText.indexOf("{");
    const jEnd    = rawText.lastIndexOf("}");
    const parsed  = JSON.parse(rawText.slice(jStart, jEnd + 1));

    return cors(200, JSON.stringify(parsed));
  } catch (err) {
    return cors(500, JSON.stringify({ error: err.message }));
  }
};

function cors(status, body) {
  return {
    statusCode: status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "content-type",
    },
    body,
  };
}
