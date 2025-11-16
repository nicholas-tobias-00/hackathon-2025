import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    const { message } = await req.json();

    if (!message) {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 }
      );
    }

    // Call Python server
    const pythonResponse = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!pythonResponse.ok) {
      const err = await pythonResponse.text();
      throw new Error(`Python server error: ${err}`);
    }

    const data = await pythonResponse.json();

    return NextResponse.json({
      response: data.response,
    });
  } catch (err) {
    console.error("Next.js API Error:", err);
    return NextResponse.json(
      { error: err.message || "Something went wrong" },
      { status: 500 }
    );
  }
}
