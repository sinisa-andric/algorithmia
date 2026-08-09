package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	spiralDynamicsPopulation = 20
	spiralDynamicsMaxSteps   = 500
	spiralDynamicsSpiralRate = 0.98
	spiralDynamicsTolerance  = 1e-6
	spiralDynamicsRangeLow   = -5.0
	spiralDynamicsRangeHigh  = 5.0
	spiralDynamicsTheta      = math.Pi / 4
)

// SpiralDynamics minimizuje konfigurisanu benchmark funkciju koristeći Spiral Dynamics Algorithm: parovi dimenzija
// rotiraju oko best-a fiksnim uglom uz kontrakciju ka centru svakog koraka, a neuparena poslednja dimenzija (ili
// 1D slučaj) koristi jednodimenzionu kontrakciju uz nezavisan šum
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func SpiralDynamics(problem models.Problem) (result models.Result, err error) {

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

	population := spiralDynamicsPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := spiralDynamicsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	spiralRate := spiralDynamicsSpiralRate
	if v, ok := problem.Payload["spiral_rate"].(float64); ok {
		spiralRate = v
	}

	tolerance := spiralDynamicsTolerance
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
			individual[d] = spiralDynamicsRangeLow + rand.Float64()*(spiralDynamicsRangeHigh-spiralDynamicsRangeLow)
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

	cosTheta := math.Cos(spiralDynamicsTheta)
	sinTheta := math.Sin(spiralDynamicsTheta)

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range individuals {
			d := 0
			for d+1 < dimensions {
				dx := individuals[i][d] - best[d]
				dy := individuals[i][d+1] - best[d+1]

				// čista rotacija+kontrakcija bez nezavisnog šuma bi bila potpuno deterministička putanja
				// određena samo inicijalnom pozicijom, pa se dodaje mali nezavisan šum da populacija ne
				// prati identičnu deterministički sinhronizovanu orbitu
				individuals[i][d] = clamp(best[d]+spiralRate*(dx*cosTheta-dy*sinTheta)+(rand.Float64()*2-1)*0.05, spiralDynamicsRangeLow, spiralDynamicsRangeHigh)
				individuals[i][d+1] = clamp(best[d+1]+spiralRate*(dx*sinTheta+dy*cosTheta)+(rand.Float64()*2-1)*0.05, spiralDynamicsRangeLow, spiralDynamicsRangeHigh)
				d += 2
			}
			if d < dimensions {
				dx := individuals[i][d] - best[d]
				individuals[i][d] = clamp(best[d]+spiralRate*dx*cosTheta+(rand.Float64()*2-1)*0.1, spiralDynamicsRangeLow, spiralDynamicsRangeHigh)
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
		Method:     "spiral_dynamics",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
