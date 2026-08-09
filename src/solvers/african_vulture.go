package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	africanVulturePopulation = 20
	africanVultureMaxSteps   = 1000
	africanVultureTolerance  = 1e-6
	africanVultureRangeLow   = -5.0
	africanVultureRangeHigh  = 5.0
)

// AfricanVulture minimizuje konfigurisanu benchmark funkciju koristeći African Vulture Optimization Algorithm:
// svaki lešinar prati jednog od dva vodeća lešinara, a nasumičan faktor koji opada tokom izvršavanja bira između
// eksploracije (direktan pristup ili kombinacija dva nasumična lešinara), spiralnog prilaženja ili direktnog homing-a
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func AfricanVulture(problem models.Problem) (result models.Result, err error) {

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

	population := africanVulturePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := africanVultureMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := africanVultureTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	vultures := make([][]float64, population)
	values := make([]float64, population)
	for i := range vultures {
		vulture := make([]float64, dimensions)
		for d := range vulture {
			vulture[d] = africanVultureRangeLow + rand.Float64()*(africanVultureRangeHigh-africanVultureRangeLow)
		}
		vultures[i] = vulture
		values[i] = fn.Evaluate(vulture)
	}

	best1, best2 := topTwo(vultures, values)
	bestValue := fn.Evaluate(best1)

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range vultures {
			leader := best2
			if rand.Float64() >= 0.5 {
				leader = best1
			}

			f := (2*rand.Float64() - 1) * (1 - float64(steps)/float64(maxSteps))
			absF := math.Abs(f)

			switch {
			case absF >= 1:
				if rand.Float64() >= 0.5 {
					for d := range vultures[i] {
						vultures[i][d] = leader[d] - (leader[d]-vultures[i][d])*rand.Float64()
					}
				} else {
					r1 := vultures[rand.IntN(population)]
					r2 := vultures[rand.IntN(population)]
					for d := range vultures[i] {
						vultures[i][d] = leader[d] - math.Abs(leader[d]-vultures[i][d]) + rand.Float64()*(r1[d]-r2[d])
					}
				}
			case absF >= 0.5:
				for d := range vultures[i] {
					vultures[i][d] = math.Abs(leader[d]-vultures[i][d])*(f+rand.Float64())*math.Cos(rand.Float64()*2*math.Pi) + leader[d]
				}
			default:
				for d := range vultures[i] {
					vultures[i][d] = leader[d] - math.Abs(leader[d]-vultures[i][d])*rand.Float64()*sign(f)
				}
			}

			for d := range vultures[i] {
				vultures[i][d] = clamp(vultures[i][d], africanVultureRangeLow, africanVultureRangeHigh)
			}

			values[i] = fn.Evaluate(vultures[i])
		}

		best1, best2 = topTwo(vultures, values)
		bestValue = fn.Evaluate(best1)

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best1, bestValue, false)
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
		trajectory = recordTrajectory(trajectory, steps, best1, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "african_vulture",
		Point:      best1,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
