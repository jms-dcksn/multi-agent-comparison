from langchain_core.messages import AIMessageChunk
from uuid_utils import uuid7

from sub_agent import agent


def main():
    print("Demo script writer. Type 'quit' to exit.\n")
    config = {"configurable": {"thread_id": str(uuid7())}}
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "quit":
            break

        print("\nAgent: ", end="", flush=True)
        for mode, payload in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, _ = payload
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    print(chunk.content, end="", flush=True)
            elif mode == "updates":
                for node, data in payload.items():
                    msg = data["messages"][-1]
                    for call in getattr(msg, "tool_calls", []) or []:
                        print(f"\n[tool call] {call['name']}({call['args']})", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
