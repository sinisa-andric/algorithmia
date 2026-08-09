package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	slimeMouldAgents    = 20
	slimeMouldMaxSteps  = 500
	slimeMouldTolerance = 1e-6
	slimeMouldSpread    = 10.0
	slimeMouldEpsilon   = 1e-10
	slimeMouldBound     = 15.0
)

// SlimeMould minimizuje sphere funkciju koristeći Slime Mould Algorithm: agenti ili konvergiraju ka najboljem preko
// razlike dva nasumična agenta ponderisane fitnesom, ili se kreću nasumično,
// zavisno od verovatnoće izvedene iz njihove fitnes udaljenosti od trenutno najboljeg
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SlimeMould(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	count := slimeMouldAgents
	if v, ok := problem.Payload["agents"].(float64); ok {
		count = int(v)
	}
	if count < 2 {
		count = 2
	}

	maxSteps := slimeMouldMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := slimeMouldTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	agents := make([][]float64, count)
	for i := range agents {
		agent := make([]float64, dimensions)
		for d := range agent {
			agent[d] = problem.Point[d] + (rand.Float64()*2-1)*slimeMouldSpread
		}
		agents[i] = agent
	}
	sortByFitness(fn, agents)

	best := append([]float64(nil), agents[0]...)
	bestValue := fn.Evaluate(best)

	globalBest := append([]float64(nil), best...)
	globalBestValue := bestValue

	half := count / 2

	steps := 0
	for ; steps < maxSteps && globalBestValue >= tolerance; steps++ {

		sortByFitness(fn, agents)

		next := make([][]float64, count)
		for i, agent := range agents {
			agentValue := fn.Evaluate(agent)

			var w float64
			if i < half {
				w = 1 + rand.Float64()*math.Log(1/(agentValue+slimeMouldEpsilon)+1)
			} else {
				w = 1 - rand.Float64()*math.Log(1/(agentValue+slimeMouldEpsilon)+1)
			}

			p := math.Tanh(math.Abs(agentValue - bestValue))

			newAgent := make([]float64, dimensions)
			if rand.Float64() < p {
				a := agents[rand.IntN(count)]
				b := agents[rand.IntN(count)]
				for d := range newAgent {
					newAgent[d] = best[d] + w*(a[d]-b[d])*rand.Float64()
				}
			} else {
				for d := range newAgent {
					newAgent[d] = best[d] + 0.05*(rand.Float64()*2-1)*best[d]
				}
			}

			for d := range newAgent {
				if newAgent[d] > slimeMouldBound {
					newAgent[d] = slimeMouldBound
				} else if newAgent[d] < -slimeMouldBound {
					newAgent[d] = -slimeMouldBound
				}
			}

			next[i] = newAgent
		}

		agents = next

		bestValue = fn.Evaluate(agents[0])
		best = append([]float64(nil), agents[0]...)
		for _, agent := range agents {
			if value := fn.Evaluate(agent); value < bestValue {
				bestValue = value
				best = append([]float64(nil), agent...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, false)
			}
		}

		if bestValue < globalBestValue {
			globalBestValue = bestValue
			globalBest = append([]float64(nil), best...)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, globalBest, globalBestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "slime_mould",
		Point:      globalBest,
		Value:      globalBestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
