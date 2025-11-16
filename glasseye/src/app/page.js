"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Plus,
  Menu,
  TrendingUp,
  Clock,
  MessageSquare,
  Zap,
  Eye,
  X,
  Loader2,
  Network,
  CloudAlert,
  CircleDollarSign,
  Earth,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";

export default function ChatInterface() {
  const [messages, setMessages] = useState([
    { id: 1, role: "assistant", content: "Hello! How can I help you today?" },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [bgColorState, setBgColorState] = useState(0); // 0: blue, 1: green, 2: red
  const [temperature, setTemperature] = useState(200); // 200 = cool blue
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);
  const [rightSidebarLoading, setRightSidebarLoading] = useState(false);
  const [selectedMessageId, setSelectedMessageId] = useState(null);
  const messagesEndRef = useRef(null);
  const [modelMetrics, setModelMetrics] = useState({
    co2_grams: 0,
    energy_kwh: 0,
    tokens: 0,
    cost: 0,
  });

  // Previous chats placeholder data
  const previousChats = [
    { id: 1, title: "React Best Practices", time: "2 hours ago" },
    { id: 2, title: "API Integration Help", time: "Yesterday" },
    { id: 3, title: "CSS Grid Layout", time: "2 days ago" },
    { id: 4, title: "Database Design", time: "3 days ago" },
    { id: 5, title: "Authentication Flow", time: "1 week ago" },
  ];

  // LangSmith metrics placeholder data
  const metrics = [
    {
      label: "Carbon Emissions (gCO₂):",
      value: modelMetrics.co2_grams.toFixed(6),
      icon: CloudAlert,
    },
    {
      label: "Energy Consumption (kWh):",
      value: modelMetrics.energy_kwh.toExponential(4),
      icon: Zap,
    },
    { label: "Total Tokens:", value: modelMetrics.tokens, icon: Earth },
    {
      label: "Cost ($):",
      value: modelMetrics.cost.toFixed(6),
      icon: CircleDollarSign,
    },
  ];

  // Dummy per-response metrics (increment each time the AI responds)
  const METRIC_INCREMENT = {
    co2_grams: 0.0025, // grams CO₂ per response
    energy_kwh: 0.000015, // kWh consumed
    tokens: 45, // assume average tokens per completion
    cost: 0.00012, // $ cost per response
  };

  // Background color transition
  // const backgroundColors = [
  //   "from-sky-50 to-blue-100", // Blue
  //   "from-emerald-50 to-green-100", // Green
  //   "from-rose-50 to-red-100", // Red
  // ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // const handleSendMessage = () => {
  //   if (!inputValue.trim()) return;

  //   const newMessage = {
  //     id: messages.length + 1,
  //     role: "user",
  //     content: inputValue,
  //   };

  //   setMessages([...messages, newMessage]);
  //   setInputValue("");

  //   // Simulate AI response
  //   setTimeout(() => {
  //     setMessages((prev) => [
  //       ...prev,
  //       {
  //         id: prev.length + 1,
  //         role: "assistant",
  //         content:
  //           "This is a simulated response. In a real implementation, this would be connected to the AI backend.",
  //       },
  //     ]);

  //     // 🌡️ global warming effect - shift toward red
  //     setTemperature((prev) => Math.max(prev - 10, 0));
  //   }, 1000);
  // };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: messages.length + 1,
      role: "user",
      content: inputValue,
    };

    setMessages([...messages, userMessage]);
    const messageToSend = inputValue;
    setInputValue("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messageToSend }),
      });

      // const data = await response.json();
      // setTemperature((prev) => Math.max(prev - 20, 0));

      // const assistantMessage = {
      //   id: messages.length + 2,
      //   role: "assistant",
      //   content: data.response,
      // };

      // setMessages((prev) => [...prev, assistantMessage]);

      const data = await response.json();

      // 🌡️ temperature effect
      setTemperature((prev) => Math.max(prev - 20, 0));

      // update UI metrics from backend
      // Accumulate metrics (dummy values per response)
      setModelMetrics((prev) => ({
        co2_grams: prev.co2_grams + METRIC_INCREMENT.co2_grams,
        energy_kwh: prev.energy_kwh + METRIC_INCREMENT.energy_kwh,
        tokens: prev.tokens + METRIC_INCREMENT.tokens,
        cost: prev.cost + METRIC_INCREMENT.cost,
      }));

      // show assistant response
      const assistantMessage = {
        id: messages.length + 2,
        role: "assistant",
        content: data.response,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error("Error:", err);
      setMessages((prev) => [
        ...prev,
        {
          id: prev.length + 1,
          role: "assistant",
          content: "⚠️ Error contacting model.",
        },
      ]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Function to change background color
  const changeBackgroundColor = () => {
    setBgColorState((prev) => (prev + 1) % 3);
  };

  // Handle view graph click
  const handleViewGraph = (messageId) => {
    setSelectedMessageId(messageId);
    setRightSidebarOpen(true);
    setRightSidebarLoading(true);

    // Simulate loading
    setTimeout(() => {
      setRightSidebarLoading(false);
    }, 1500);
  };

  return (
    <div
      className="flex h-screen overflow-hidden transition-all duration-700"
      style={{
        background: `linear-gradient(
        to bottom right,
        hsl(${temperature}, 90%, 88%),
        hsl(${Math.max(temperature - 40, 0)}, 85%, 80%)
        )`,
      }}
    >
      {/* Left Sidebar */}
      <div
        className={`${
          sidebarOpen ? "w-64" : "w-0"
        } transition-all duration-300 bg-white border-r border-gray-200 flex flex-col overflow-hidden`}
      >
        {/* Top Half - LangSmith Metrics */}
        <div className="h-1/2 border-b border-gray-200 p-4 overflow-auto">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-700">
              LangSmith Metrics
            </h2>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={changeBackgroundColor}
              title="Change background color"
            >
              <TrendingUp className="h-4 w-4" />
            </Button>
          </div>
          <div className="space-y-3">
            {metrics.map((metric, idx) => (
              <div
                key={idx}
                className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-3 border border-gray-200"
              >
                <div className="flex items-center gap-2 mb-1">
                  <metric.icon className="h-3.5 w-3.5 text-gray-500" />
                  <span className="text-xs text-gray-600">{metric.label}</span>
                </div>
                <div className="text-lg font-semibold text-gray-900">
                  {metric.value}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Half - Previous Chats */}
        <div className="h-1/2 p-4 overflow-auto">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-700">
              Previous Chats
            </h2>
            <Button variant="ghost" size="icon" className="h-6 w-6">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
          <ScrollArea className="h-full">
            <div className="space-y-2">
              {previousChats.map((chat) => (
                <button
                  key={chat.id}
                  className="w-full text-left p-3 rounded-lg hover:bg-gray-100 transition-colors group"
                >
                  <div className="flex items-start gap-2">
                    <MessageSquare className="h-4 w-4 text-gray-400 mt-0.5 group-hover:text-gray-600" />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-gray-900 truncate">
                        {chat.title}
                      </div>
                      <div className="text-xs text-gray-500">{chat.time}</div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </ScrollArea>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200 px-4 py-3 flex items-center gap-3 flex-shrink-0">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="h-8 w-8"
          >
            <Menu className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold text-gray-900">GlassEye</h1>
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-md">
              <Eye className="h-4 w-4 text-white" />
            </div>
          </div>
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Messages Container with Glassmorphism Chatbox */}
          <div className="flex-1 flex items-center justify-center px-4 py-8">
            <div
              className={`w-full ${
                rightSidebarOpen ? "max-w-3xl" : "max-w-4xl"
              } h-full transition-all duration-300`}
            >
              {/* Glassmorphism Chatbox with Internal Scroll */}
              <div className="bg-white/40 backdrop-blur-xl rounded-3xl border border-white/50 shadow-2xl h-full relative overflow-hidden flex flex-col">
                {/* Subtle highlight effects */}
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-white/60 to-transparent"></div>
                <div className="absolute top-0 left-1/4 w-32 h-32 bg-blue-400/20 rounded-full blur-3xl"></div>
                <div className="absolute bottom-0 right-1/4 w-40 h-40 bg-purple-400/20 rounded-full blur-3xl"></div>

                {/* Messages with Internal ScrollArea */}
                <ScrollArea className="flex-1 p-6 h-full">
                  <div className="relative z-10 space-y-6">
                    {messages.map((message) => (
                      <div key={message.id}>
                        <div
                          className={`flex ${
                            message.role === "user"
                              ? "justify-end"
                              : "justify-start"
                          }`}
                        >
                          <div
                            className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                              message.role === "user"
                                ? "bg-gray-900/90 backdrop-blur-sm text-white shadow-lg"
                                : "bg-white/80 backdrop-blur-md text-gray-900 border border-white/60 shadow-lg"
                            }`}
                          >
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">
                              {message.content}
                            </p>
                          </div>
                        </div>
                        {message.role === "assistant" && (
                          <div className="flex justify-start mt-2 ml-2">
                            <button
                              onClick={() => handleViewGraph(message.id)}
                              className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1 hover:underline transition-all"
                            >
                              <Network className="h-3 w-3" />
                              View reasoning graph
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>
              </div>
            </div>
          </div>

          {/* Right Sidebar */}
          <div
            className={`${
              rightSidebarOpen ? "w-96" : "w-0"
            } transition-all duration-300 bg-white/90 backdrop-blur-md border-l border-gray-200 flex flex-col overflow-hidden shadow-2xl`}
          >
            {rightSidebarOpen && (
              <>
                {/* Right Sidebar Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                  <h2 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                    <Network className="h-4 w-4 text-blue-600" />
                    Reasoning Graph
                  </h2>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setRightSidebarOpen(false)}
                    className="h-7 w-7"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>

                {/* Right Sidebar Content */}
                <div className="flex-1 p-6 overflow-auto">
                  {rightSidebarLoading ? (
                    <div className="flex flex-col items-center justify-center h-full space-y-4">
                      <Loader2 className="h-12 w-12 text-blue-600 animate-spin" />
                      <p className="text-sm text-gray-600">
                        Loading reasoning graph...
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {/* Placeholder Graph Structure */}
                      <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6 border border-blue-200">
                        <div className="space-y-4">
                          {/* Node 1 */}
                          <div className="bg-white rounded-lg p-4 shadow-md border-l-4 border-blue-500">
                            <div className="flex items-center gap-2 mb-2">
                              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                              <span className="text-xs font-semibold text-gray-900">
                                Initial Query
                              </span>
                            </div>
                            <p className="text-xs text-gray-600">
                              User input processed by GlassEye
                            </p>
                          </div>

                          {/* Connection Line */}
                          <div className="flex justify-center">
                            <div className="w-0.5 h-8 bg-gradient-to-b from-blue-300 to-purple-300"></div>
                          </div>

                          {/* Node 2 */}
                          <div className="bg-white rounded-lg p-4 shadow-md border-l-4 border-purple-500">
                            <div className="flex items-center gap-2 mb-2">
                              <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                              <span className="text-xs font-semibold text-gray-900">
                                Choosing Foundation Model
                              </span>
                            </div>
                            <p className="text-xs text-gray-600">
                              Use [MODEL NAME] for response generation
                            </p>
                          </div>

                          {/* Connection Line */}
                          <div className="flex justify-center">
                            <div className="w-0.5 h-8 bg-gradient-to-b from-purple-300 to-green-300"></div>
                          </div>

                          {/* Node 3 */}
                          <div className="bg-white rounded-lg p-4 shadow-md border-l-4 border-green-500">
                            <div className="flex items-center gap-2 mb-2">
                              <div className="w-3 h-3 rounded-full bg-green-500"></div>
                              <span className="text-xs font-semibold text-gray-900">
                                Response Generation
                              </span>
                            </div>
                            <p className="text-xs text-gray-600">
                              Generating final response
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Additional Info */}
                      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                        <h3 className="text-xs font-semibold text-gray-900 mb-2">
                          Graph Statistics
                        </h3>
                        <div className="space-y-2">
                          <div className="flex justify-between text-xs">
                            <span className="text-gray-600">Total Nodes:</span>
                            <span className="font-medium text-gray-900">3</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-gray-600">
                              Processing Time:
                            </span>
                            <span className="font-medium text-gray-900">
                              1.2s
                            </span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-gray-600">Confidence:</span>
                            <span className="font-medium text-green-600">
                              94%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 bg-white/80 backdrop-blur-sm flex-shrink-0">
          <div className="max-w-3xl mx-auto px-4 py-4">
            <div className="relative flex items-center gap-2 bg-white/90 backdrop-blur-md rounded-xl border border-white/60 shadow-lg focus-within:border-blue-400 focus-within:shadow-xl transition-all">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Message GlassEye..."
                className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent px-4"
              />
              <Button
                onClick={handleSendMessage}
                disabled={!inputValue.trim()}
                size="icon"
                className="h-9 w-9 mr-2 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:bg-gray-200 shadow-md"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-gray-600 text-center mt-2">
              AI can make mistakes. Consider checking important information.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
