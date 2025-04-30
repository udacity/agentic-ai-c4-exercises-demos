diagram_descriptions = [{'what_is_x':"""
        [DIAGRAM: A flowchart showing a multi-agent system. A user connects to a coordinator agent with a 
        bidirectional "request/response" arrow. The coordinator has a decision loop that branches to either 
        use tools directly or delegate to specialized agents. These specialized agents (shown as 3-4 separate boxes)
         have access to various tools (database, search, code execution). Arrows show communication flow between components.]
"""}, 
                        {'module_1':""" 
                                 [DIAGRAM: An architecture blueprint showing common multi-agent patterns. The left side shows an "Orchestrator-Worker" 
        pattern with a central manager connected to multiple specialized workers. The right side shows a "Peer-to-Peer" pattern where
        specialized agents connect directly to each other in a mesh. In the former, all communication flows through a central agent, whereas in the latter, communication 
        begins directly with individual agents who can decide to pass things along to others in the network.
        Arrows indicate message passing between components. 
        Labels highlight key design considerations like "Role Definition", "Communication Protocol", and "State Management".]

                         """
                         }, 
                        {'module_1_exercise_solution': """
                [DIAGRAM: A flowchart showing the helpdesk multi-agent system with four connected components.
                The Orchestrator Agent is at the center with arrows connecting to three specialized agents: 
                Classification Agent (top), Knowledge Agent (right), and Response Agent (bottom). 
                Arrows show the data flow between components, with user input entering the Orchestrator and the final 
                response coming from the Response Agent.]

                         """}, 
                        {'module_2':"""
                         
        [DIAGRAM: A code structure visualization showing agent implementation. The left side shows "Agent Definition" 
        with a base agent class and specialized subclasses. The middle shows "Communication" with message objects and 
        interface definitions. The right shows "Tools Integration" with specialized tools assigned to different agents. 
        Colored connections show relationships between components.]

                         """
                         },
                        {'module_2_exercise_solution':"""
                         
        [DIAGRAM: A flowchart showing three connected agent functions. The finder_agent (left) connects to 
        curator_agent (middle), which connects to formatter_agent (right), with data flowing left to right.
        The process_request function wraps around all three, showing the orchestration flow.]
                         """
                         },
                        {'module_3':"""
                         
        [DIAGRAM: An orchestration flow chart showing a central "Orchestrator" node connected to multiple worker agents.
        The diagram shows three key orchestration patterns: sequential execution (agents in a chain), parallel execution (multiple agents working simultaneously), and conditional branching (decision points that route to different agents). Arrows show message flow between components, with state data being passed back to the orchestrator after each agent completes its task.]

                            """
                            },
                            {'module_3_exercise_solution':"""
                             
        [DIAGRAM: A flow chart showing a sequential workflow with three specialized functions. 
        Starting with "diagnose" (analyzes customer issue), flowing to "find_solution" (retrieves appropriate fix), and finally to "verify_solution" (confirms effectiveness). A "handle_support_request" function wraps around all three, showing orchestration.]
                            """
                            },
                            {'module_4':"""
                                     [DIAGRAM: A data flow diagram showing message routing between agents. The central 
        component is a "Router" that contains decision logic. Multiple message types (shown 
        as different colored envelopes) flow into the router, which examines properties and 
        directs them to appropriate agent endpoints. Decision nodes show different routing 
        patterns: content-based routing, round-robin distribution, and priority-based queuing. 
        Arrows show message flow with metadata labels attached to each path.]
                            """
                            },
                            {'module_4_exercise_solution':"""
                                        [DIAGRAM: A flowchart showing a message routing system with three components:
        "Message Queue" (top), "Router" (middle), and "Agent Endpoints" (bottom).
        The Message Queue receives messages and passes them to the Router, which uses decision logic to route messages
        to the appropriate Agent Endpoints. The diagram shows different routing patterns:
        content-based routing (based on message type), round-robin distribution (equal load balancing), and priority-based queuing (urgent messages first).
        Arrows indicate the flow of messages and decisions made at each step.]
                            """
                            },
                            
                            {'module_5':"""   
                                     [DIAGRAM: A comprehensive state management diagram with two main sections. The left shows "Conversation State" 
        (ephemeral) with a chat history and short-term memory component. The right shows "System State" (persistent) 
        with a database connection. The middle shows state transitions between conversations, including save/load operations 
        and recovery points. Arrows indicate how state flows between components and across sessions. 
        Color coding distinguishes between temporary and persistent data types.]
                            """
                            },
                            {'module_5_exercise_solution':"""
        [DIAGRAM: A flow chart showing state management in a tutoring system. At the top is a "user_states" database. 
        Three main functions connect to it: save_state (writes data), load_state (reads data), and 
        update_progress (reads, modifies, writes). Two session functions, start_session and end_session, 
        act as wrappers around the core functions. Arrows show the data flow between components.]

                            """
                            },
                            {'module_6':"""
                                     [DIAGRAM: A state coordination system showing multiple agents and a central coordination mechanism. 
        The diagram features agent states as colored boxes (each with internal state variables), 
        connected to a central "State Broker" that manages synchronization. 
        Dotted lines show state update messages flowing between components. 
        A "Conflict Detection" module highlights areas where agent states conflict, 
        and a "Resolution Logic" component shows how conflicts are addressed. A timeline at the 
        bottom illustrates how states stay synchronized across time steps in the system.]
                            """
                            },
                            {'module_7':"""
                                     [DIAGRAM: A multi-agent RAG architecture showing specialized retrieval agents on the 
        left (Scientific Literature Agent, News Agent, Database Agent, and Web Search Agent), 
        each connected to its own data source. The middle shows a "Retrieval Coordinator" 
        that manages requests to these agents. On the right is a "Synthesis Agent" that combines 
        information from multiple sources. Arrows show how user queries flow through the system, 
        with dotted lines representing optional follow-up queries when information gaps are detected. 
        A "Knowledge Gap Analyzer" component sits between the coordinator and synthesis agent to identify missing information.]
                            """
                            },
]


from npcpy.llm_funcs import generate_image, get_llm_response


for description in diagram_descriptions:
    for key, value in description.items():
        better_prompt = get_llm_response('''
                                         Please make this prompt better for getting a more successful image generation promptT:
                                         "
                                         
                                         '''+ value + ' PLEASE USE A TRANSPARENT BACKGROUND IN GENERATING YOUR DIAGRAMS. "  In your response, only reply directly with the prompt to use without any additional text or explanation.',                                         
        model='chatgpt-4o-latest', 
        provider='openai')['response']
        
        print(f"Better prompt for {key}: {better_prompt}")
        
        print(f"Generating image for {key}...")
        # write the new prompts to a csv
        with open('diagram_prompts.csv', 'a') as f:
            f.write(f"{key},{better_prompt}\n")
            
        generate_image(better_prompt,  model='gpt-image-1', provider='openai', filename=f"{key}.png", height=1024, width=1024)
        print(f"Image saved as {key}.png")
        
        