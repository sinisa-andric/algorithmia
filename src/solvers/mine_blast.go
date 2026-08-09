package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	mineBlastPopulation = 20
	mineBlastMaxSteps   = 500
	mineBlastTolerance  = 1e-6
	mineBlastRangeLow   = -5.0
	mineBlastRangeHigh  = 5.0
)

// MineBlast minimizuje konfigurisanu benchmark funkciju koristeći Mine Blast Algorithm: udarni talas eksplozije
// gura svaki fragment ka best-u brzinom koja opada tokom izvršavanja i slabi sa udaljenošću od best-a, uz šum koji
// predstavlja nepravilnu putanju fragmenta
// problem.Point inicijalizuje populaciju fragmenata i određuje njenu dimenzionalnost
func MineBlast(problem models.Problem) (result models.Result, err error) {

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

	population := mineBlastPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := mineBlastMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := mineBlastTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = mineBlastRangeLow + rand.Float64()*(mineBlastRangeHigh-mineBlastRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		shockWaveSpeed := 2.0*(1-float64(steps)/float64(maxSteps)) + 0.1

		for i := range individuals {
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = individuals[i][d] - best[d]
			}
			distanceFactor := 1 / (1 + norm(diff))

			for d := range individuals[i] {
				direction := 1.0
				if best[d] < individuals[i][d] {
					direction = -1.0
				} else if best[d] == individuals[i][d] {
					direction = 0.0
				}
				individuals[i][d] = clamp(individuals[i][d]+shockWaveSpeed*rand.Float64()*distanceFactor*direction+(rand.Float64()*2-1)*0.2, mineBlastRangeLow, mineBlastRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "mine_blast",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
