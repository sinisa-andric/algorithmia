package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	mayflyPopulation = 20
	mayflyMaxSteps   = 1000
	mayflyTolerance  = 1e-6
	mayflyRangeLow   = -5.0
	mayflyRangeHigh  = 5.0
)

// Mayfly minimizuje konfigurisanu benchmark funkciju koristeći Mayfly Algorithm: mužjaci se okupljaju oko globalno
// najboljeg mužjaka, dok se ženke ili privlače ka boljem upareniom mužjaku ili nasumično lutaju,
// sa vidljivošću koja opada sa udaljenošću
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Mayfly(problem models.Problem) (result models.Result, err error) {

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

	population := mayflyPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := mayflyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := mayflyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	half := max(1, population/2)

	males := make([][]float64, half)
	females := make([][]float64, half)
	maleValues := make([]float64, half)
	femaleValues := make([]float64, half)
	velocityM := make([][]float64, half)
	velocityF := make([][]float64, half)

	for i := range half {
		male := make([]float64, dimensions)
		female := make([]float64, dimensions)
		for d := range dimensions {
			male[d] = mayflyRangeLow + rand.Float64()*(mayflyRangeHigh-mayflyRangeLow)
			female[d] = mayflyRangeLow + rand.Float64()*(mayflyRangeHigh-mayflyRangeLow)
		}
		males[i] = male
		females[i] = female
		maleValues[i] = fn.Evaluate(male)
		femaleValues[i] = fn.Evaluate(female)
		velocityM[i] = make([]float64, dimensions)
		velocityF[i] = make([]float64, dimensions)
	}

	bestMale := append([]float64(nil), males[0]...)
	bestMaleValue := maleValues[0]
	for i, v := range maleValues {
		if v < bestMaleValue {
			bestMaleValue = v
			bestMale = append([]float64(nil), males[i]...)
		}
	}

	best := append([]float64(nil), bestMale...)
	bestValue := bestMaleValue

	g1 := 1.0
	g2 := 1.5
	beta := 2.0

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for i := range males {
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = bestMale[d] - males[i][d]
			}
			r := norm(diff)
			vis := math.Exp(-beta * r)

			for d := range males[i] {
				velocityM[i][d] = 0.9*velocityM[i][d] + g1*vis*(bestMale[d]-males[i][d]) + g2*vis*(bestMale[d]-males[i][d])
				males[i][d] = males[i][d] + velocityM[i][d]
				males[i][d] = clamp(males[i][d], mayflyRangeLow, mayflyRangeHigh)
			}
			maleValues[i] = fn.Evaluate(males[i])
		}

		for i := range females {
			bestM := males[0]
			diff := make([]float64, dimensions)
			for d := range diff {
				diff[d] = bestM[d] - females[i][d]
			}
			r := norm(diff)

			pairedMaleValue := maleValues[i%len(males)]
			if pairedMaleValue < femaleValues[i] {
				for d := range females[i] {
					velocityF[i][d] = 0.9*velocityF[i][d] + g2*math.Exp(-beta*r)*(bestM[d]-females[i][d])
				}
			} else {
				for d := range females[i] {
					velocityF[i][d] = 0.9*velocityF[i][d] + rand.Float64()
				}
			}

			for d := range females[i] {
				females[i][d] = females[i][d] + velocityF[i][d]
				females[i][d] = clamp(females[i][d], mayflyRangeLow, mayflyRangeHigh)
			}
			femaleValues[i] = fn.Evaluate(females[i])
		}

		bestMale = append([]float64(nil), males[0]...)
		bestMaleValue = maleValues[0]
		for i, v := range maleValues {
			if v < bestMaleValue {
				bestMaleValue = v
				bestMale = append([]float64(nil), males[i]...)
			}
		}

		bestValue = bestMaleValue
		best = append([]float64(nil), bestMale...)
		for i, v := range femaleValues {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), females[i]...)
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
		Method:     "mayfly",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
