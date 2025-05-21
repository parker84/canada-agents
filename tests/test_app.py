from typing import Optional
from agno.agent import Agent
from agno.eval.accuracy import AccuracyEval, AccuracyResult
from agno.models.openai import OpenAIChat
from agno.tools.calculator import CalculatorTools
from agno.eval.performance import PerformanceEval
from agno.eval.reliability import ReliabilityEval, ReliabilityResult
from agno.run.response import RunResponse

def test_accuracy():
    evaluation = AccuracyEval(
        agent=Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[CalculatorTools(add=True, multiply=True, exponentiate=True)],
        ),
        prompt="What is 10*5 then to the power of 2? do it step by step",
        expected_answer="2500",
        num_iterations=1
    )
    result: Optional[AccuracyResult] = evaluation.run(print_results=True)

    assert result is not None, "Evaluation result should not be None"
    assert result.avg_score >= 8, f"Expected average score >= 8, got {result.avg_score}"

def test_performance():
    def simple_response():
        agent = Agent(model=OpenAIChat(id='gpt-4o-mini'), system_message='Be concise, reply with one sentence.', add_history_to_messages=True)
        response_1 = agent.run('What is the capital of France?')
        print(response_1.content)
        response_2 = agent.run('How many people live there?')
        print(response_2.content)
        return response_2.content
    
    simple_response_perf = PerformanceEval(
        func=simple_response, 
        num_iterations=1, 
        warmup_runs=0
    )
    result = simple_response_perf.run(print_results=True)
    assert result is not None, "Performance evaluation should return results"
    assert result.avg_run_time < 5, f"Expected average run time < 5, got {result.avg_run_time}"

def test_reliability():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[CalculatorTools(add=True, multiply=True, exponentiate=True)],
    )
    response: RunResponse = agent.run("What is 10*5 then to the power of 2? do it step by step")
    evaluation = ReliabilityEval(
        agent_response=response,
        expected_tool_calls=["multiply", "exponentiate"],
    )
    result: Optional[ReliabilityResult] = evaluation.run(print_results=True)
    assert result is not None, "Reliability evaluation should return results"
    result.assert_passed()