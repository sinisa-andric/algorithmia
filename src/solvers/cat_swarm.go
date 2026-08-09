package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	catSwarmPopulation = 20
	catSwarmMaxSteps   = 1000
	catSwarmTolerance  = 1e-6
	catSwarmRangeLow   = -5.0
	catSwarmRangeHigh  = 5.0
	catSwarmMR         = 0.1
)

// CatSwarm minimizuje konfigurisanu benchmark funkciju koristeći Cat Swarm Optimization: mačka je u modu traženja
// (nasumične mutacije kopija, biranje najbolje) sa verovatnoćom mr, inače je u modu praćenja (kretanje pod uticajem
// brzine ka najboljem rešenju)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func CatSwarm(problem models.Problem) (result models.Result, err error) {

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

	population := catSwarmPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := catSwarmMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := catSwarmTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	mr := catSwarmMR
	if v, ok := problem.Payload["mr"].(float64); ok {
		mr = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	cats := make([][]float64, population)
	velocity := make([][]float64, population)
	values := make([]float64, population)
	for i := range cats {
		cat := make([]float64, dimensions)
		for d := range cat {
			cat[d] = catSwarmRangeLow + rand.Float64()*(catSwarmRangeHigh-catSwarmRangeLow)
		}
		cats[i] = cat
		velocity[i] = make([]float64, dimensions)
		values[i] = fn.Evaluate(cat)
	}

	best := append([]float64(nil), cats[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), cats[i]...)
		}
	}

	const smp = 3
	const c1 = 2.0

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range cats {
			if rand.Float64() < mr {
				bestCopy := append([]float64(nil), cats[i]...)
				bestCopyValue := values[i]
				for range smp {
					copyPos := make([]float64, dimensions)
					for d := range copyPos {
						copyPos[d] = clamp(cats[i][d]*(1+(rand.Float64()*2-1)*0.1), catSwarmRangeLow, catSwarmRangeHigh)
					}
					copyValue := fn.Evaluate(copyPos)
					if copyValue < bestCopyValue {
						bestCopyValue = copyValue
						bestCopy = copyPos
					}
				}
				cats[i] = bestCopy
				values[i] = bestCopyValue
			} else {
				for d := range cats[i] {
					velocity[i][d] = velocity[i][d] + c1*rand.Float64()*(best[d]-cats[i][d])
					velocity[i][d] = clamp(velocity[i][d], -1.0, 1.0)
					cats[i][d] = clamp(cats[i][d]+velocity[i][d], catSwarmRangeLow, catSwarmRangeHigh)
				}
				values[i] = fn.Evaluate(cats[i])
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), cats[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "cat_swarm",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
