package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	douglasRachfordMaxSteps     = 1000
	douglasRachfordLearningRate = 0.1
	douglasRachfordLambda       = 0.01
	douglasRachfordTolerance    = 1e-6
	douglasRachfordStepClip     = 1.0
)

// DouglasRachford minimizuje konfigurisanu benchmark funkciju uz L1 penal koristeći Douglas-Rachford splitting:
// naizmenična refleksija oko glatkog člana f i L1 člana g, sa primarnom tačkom postavljenom na sredinu između dva
// stadijuma refleksije svakog koraka (umesto akumulirane korekcije, koja bi konvergirala ka pogrešnoj fiksnoj tački)
// problem.Point je početna tačka pretrage
func DouglasRachford(problem models.Problem) (result models.Result, err error) {

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

	maxSteps := douglasRachfordMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	learningRate := douglasRachfordLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	lambda := douglasRachfordLambda
	if v, ok := problem.Payload["lambda"].(float64); ok {
		lambda = v
	}

	tolerance := douglasRachfordTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	objective := func(p []float64) float64 { return fn.Evaluate(p) + lambda*l1Norm(p) }

	x := append([]float64(nil), problem.Point...)
	z := append([]float64(nil), problem.Point...)

	best := append([]float64(nil), x...)
	bestValue := objective(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, z, proximalGradEps)
		zHalf := make([]float64, dimensions)
		for d := range zHalf {
			zHalf[d] = z[d] - learningRate*grad[d]
		}

		reflected := make([]float64, dimensions)
		for d := range reflected {
			reflected[d] = 2*zHalf[d] - z[d]
		}
		zNew := proxL1(reflected, learningRate*lambda)

		// x kao akumulirana suma "0.5*(zNew-zHalf)" konvergira ka x0 - 0.5*(ukupan put z-a), što je za sphere
		// bilo tačno na pola puta do optimuma (izmereno: x zaglavljen na x0/2 čak i kada je z u potpunosti
		// konvergirao na 0). Ispravno DR rešenje je direktno postavljanje x-a na sredinu dva stadijuma
		// refleksije svakog koraka, ne akumulacija razlika kroz vreme
		deltaX := make([]float64, dimensions)
		for d := range deltaX {
			xTarget := 0.5 * (zHalf[d] + zNew[d])
			deltaX[d] = xTarget - x[d]
		}
		deltaX = clipStep(deltaX, douglasRachfordStepClip)
		for d := range x {
			x[d] += deltaX[d]
		}

		deltaZ := make([]float64, dimensions)
		for d := range deltaZ {
			deltaZ[d] = zNew[d] - zHalf[d]
		}
		deltaZ = clipStep(deltaZ, douglasRachfordStepClip)
		for d := range z {
			z[d] += deltaZ[d]
		}

		value := objective(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), false)
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
		Method:     "douglas_rachford",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
