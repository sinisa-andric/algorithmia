package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	torczonMultidirectionalMaxSteps          = 1000
	torczonMultidirectionalTolerance         = 1e-6
	torczonMultidirectionalExpansionFactor   = 2.0
	torczonMultidirectionalContractionFactor = 0.5
	torczonMultidirectionalStepClip          = 1.0
)

// torczonClipDelta ograničava pomeraj newP u odnosu na oldP na ±bound po dimenziji pre primene
func torczonClipDelta(newP, oldP []float64, bound float64) []float64 {

	delta := make([]float64, len(newP))
	for d := range delta {
		delta[d] = newP[d] - oldP[d]
	}
	delta = clipStep(delta, bound)
	result := make([]float64, len(newP))
	for d := range result {
		result[d] = oldP[d] + delta[d]
	}

	return result
}

// TorczonMultidirectional minimizuje konfigurisanu benchmark funkciju koristeći Torczon-ov multidirekcioni metod
// pretrage: umesto refleksije JEDNE najgore tačke (kao nelder_mead.go), reflektuje SVE ostale tačke simplexa
// ISTOVREMENO kroz najbolju tačku, zatim bira ekspanziju ili kontrakciju na osnovu toga da li je najbolji od
// reflektovanih skupova poboljšao rezultat
// problem.Point je početna tačka pretrage
func TorczonMultidirectional(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := torczonMultidirectionalMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := torczonMultidirectionalTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	expansionFactor := torczonMultidirectionalExpansionFactor
	if v, ok := problem.Payload["expansion_factor"].(float64); ok {
		expansionFactor = v
	}

	contractionFactor := torczonMultidirectionalContractionFactor
	if v, ok := problem.Payload["contraction_factor"].(float64); ok {
		contractionFactor = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	points := make([][]float64, dimensions+1)
	points[0] = append([]float64(nil), problem.Point...)
	for i := 1; i <= dimensions; i++ {
		p := append([]float64(nil), problem.Point...)
		p[i-1] += 1.0
		points[i] = p
	}

	values := make([]float64, len(points))
	for i := range points {
		values[i] = fn.Evaluate(points[i])
	}

	best := append([]float64(nil), points[0]...)
	bestValue := values[0]
	for i := 1; i < len(points); i++ {
		if values[i] < bestValue {
			bestValue = values[i]
			best = append([]float64(nil), points[i]...)
		}
	}

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		bestIdx := 0
		for i := 1; i < len(points); i++ {
			if values[i] < values[bestIdx] {
				bestIdx = i
			}
		}
		bestPoint := points[bestIdx]
		bestPointValue := values[bestIdx]

		otherIdx := make([]int, 0, dimensions)
		for i := range points {
			if i != bestIdx {
				otherIdx = append(otherIdx, i)
			}
		}

		reflected := make([][]float64, len(otherIdx))
		reflectedValues := make([]float64, len(otherIdx))
		reflectedBest := math.Inf(1)
		for k, idx := range otherIdx {
			raw := make([]float64, dimensions)
			for d := range raw {
				raw[d] = bestPoint[d] - (points[idx][d] - bestPoint[d])
			}
			reflected[k] = torczonClipDelta(raw, points[idx], torczonMultidirectionalStepClip)
			reflectedValues[k] = fn.Evaluate(reflected[k])
			if reflectedValues[k] < reflectedBest {
				reflectedBest = reflectedValues[k]
			}
		}

		var accepted [][]float64
		var acceptedValues []float64
		if reflectedBest < bestPointValue {
			expanded := make([][]float64, len(otherIdx))
			expandedValues := make([]float64, len(otherIdx))
			expandedBest := math.Inf(1)
			for k, idx := range otherIdx {
				raw := make([]float64, dimensions)
				for d := range raw {
					raw[d] = bestPoint[d] - expansionFactor*(points[idx][d]-bestPoint[d])
				}
				expanded[k] = torczonClipDelta(raw, points[idx], torczonMultidirectionalStepClip)
				expandedValues[k] = fn.Evaluate(expanded[k])
				if expandedValues[k] < expandedBest {
					expandedBest = expandedValues[k]
				}
			}
			if expandedBest < reflectedBest {
				accepted, acceptedValues = expanded, expandedValues
			} else {
				accepted, acceptedValues = reflected, reflectedValues
			}
		} else {
			contracted := make([][]float64, len(otherIdx))
			contractedValues := make([]float64, len(otherIdx))
			for k, idx := range otherIdx {
				raw := make([]float64, dimensions)
				for d := range raw {
					raw[d] = bestPoint[d] + contractionFactor*(points[idx][d]-bestPoint[d])
				}
				contracted[k] = torczonClipDelta(raw, points[idx], torczonMultidirectionalStepClip)
				contractedValues[k] = fn.Evaluate(contracted[k])
			}
			accepted, acceptedValues = contracted, contractedValues
		}

		for k, idx := range otherIdx {
			points[idx] = accepted[k]
			values[idx] = acceptedValues[k]
		}

		for i := range points {
			if values[i] < bestValue {
				bestValue = values[i]
				best = append([]float64(nil), points[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
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
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "torczon_multidirectional",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
