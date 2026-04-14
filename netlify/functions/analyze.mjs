/**
 * Netlify Function: /api/analyze
 *
 * Proxies Claude vision requests for pre-processing scene analysis.
 * The Anthropic API key is either taken from the ANTHROPIC_API_KEY
 * environment variable (site-owner pays) or passed in the request body
 * (user brings their own key).
 *
 * POST body (JSON):
 *   imageBase64  – base64-encoded JPEG/PNG (max ~1024px recommended)
 *   mediaType    – "image/jpeg" | "image/png"
 *   apiKey       – user's Anthropic key (ignored if env var is set)
 *   model        – Claude model slug
 */

const SYSTEM = `You are a 3D reconstruction expert. Analyse the photograph and return ONLY valid JSON with this exact structure:
{
  "scene_summary": "one sentence describing the scene",
  "main_objects": ["object1", "object2"],
  "lighting_conditions": "description",
  "geometry_complexity": "simple|moderate|complex",
  "depth_confidence": "high|medium|low",
  "preprocessing_advice": "one tip for better 3D quality",
  "optimal_viewing_angles": ["front", "left", "top"],
  "tour_suggestions": ["tip1", "tip2", "tip3"]
}`;

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

  const { imageBase64, mediaType = "image/jpeg", apiKey, model = "claude-sonnet-4-5" } = body;

  const key = process.env.ANTHROPIC_API_KEY || apiKey;
  if (!key) {
    return cors(400, JSON.stringify({ error: "No Anthropic API key provided." }));
  }
  if (!imageBase64) {
    return cors(400, JSON.stringify({ error: "imageBase64 is required." }));
  }

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
        max_tokens: 2048,
        system: SYSTEM,
        messages: [{
          role: "user",
          content: [
            {
              type: "image",
              source: { type: "base64", media_type: mediaType, data: imageBase64 },
            },
            {
              type: "text",
              text: "Analyse this photograph for 3D reconstruction. Return only the JSON object.",
            },
          ],
        }],
      }),
    });

    if (!res.ok) {
      const err = await res.text().catch(() => "");
      return cors(res.status, JSON.stringify({ error: `Anthropic error ${res.status}`, detail: err.slice(0, 300) }));
    }

    const data = await res.json();
    // Extract the text block and parse JSON
    const rawText = data.content?.[0]?.text || "{}";
    const jsonStart = rawText.indexOf("{");
    const jsonEnd   = rawText.lastIndexOf("}");
    const parsed    = JSON.parse(rawText.slice(jsonStart, jsonEnd + 1));

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
