package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	sheepPopulation = 20
	sheepMaxSteps   = 1000
	sheepTolerance  = 1e-6
	sheepRangeLow   = -5.0
	sheepRangeHigh  = 5.0
)

// Sheep minimizuje konfigurisanu benchmark funkciju koristeći Sheep Flock Optimization: svaka ovca kombinuje
// stadno privlačenje ka centru mase celog stada i privlačenje ka najboljem pronađenom rešenju
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Sheep(problem models.Problem) (result models.Result, err error) {

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

	population := sheepPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := sheepMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := sheepTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	sheepFlock := make([][]float64, population)
	values := make([]float64, population)
	for i := range sheepFlock {
		sheep := make([]float64, dimensions)
		for d := range sheep {
			sheep[d] = sheepRangeLow + rand.Float64()*(sheepRangeHigh-sheepRangeLow)
		}
		sheepFlock[i] = sheep
		values[i] = fn.Evaluate(sheep)
	}

	best := append([]float64(nil), sheepFlock[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), sheepFlock[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		center := make([]float64, dimensions)
		for _, s := range sheepFlock {
			for d := range center {
				center[d] += s[d]
			}
		}
		for d := range center {
			center[d] /= float64(population)
		}

		for i := range sheepFlock {
			for d := range sheepFlock[i] {
				// Bez nasumične perturbacije, čim se centar i best približe roj gubi diverzitet i
				// zamrzava se oko trenutnog best-a i pre nego što stigne blizu pravog minimuma
				sheepFlock[i][d] = sheepFlock[i][d] + rand.Float64()*(center[d]-sheepFlock[i][d]) + rand.Float64()*(best[d]-sheepFlock[i][d]) + (rand.Float64()*2-1)*0.2*(1-float64(steps)/float64(maxSteps))
				sheepFlock[i][d] = clamp(sheepFlock[i][d], sheepRangeLow, sheepRangeHigh)
			}
			values[i] = fn.Evaluate(sheepFlock[i])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), sheepFlock[i]...)
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
		Method:     "sheep",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
