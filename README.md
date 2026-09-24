# AI-Based Predictive Auto-Scaling and Resource Optimization for Hybrid-Cloud Infrastructure

## Project Overview

This project investigates an AI-based predictive resource-management framework for hybrid-cloud infrastructure.

The system aims to use historical application and infrastructure telemetry to predict future workload and resource requirements, make scaling decisions, and determine suitable workload placement across private and public cloud environments.

The project focuses on improving resource utilization while maintaining application performance, Service-Level Objectives (SLOs), and operational cost efficiency.

## Research Problem

Traditional reactive autoscaling responds to workload changes after they occur. This can lead to scaling delays, performance degradation, unnecessary resource allocation, and increased operational cost.

This project investigates whether future workload and resource information can be used to coordinate:

- Predictive workload/resource forecasting
- Resource scaling
- Resource allocation
- Private/public cloud placement
- SLO monitoring
- Cost-aware decision making
- Continuous feedback and adaptation

## Research Gap

Existing research has investigated predictive autoscaling, Kubernetes resource management, hybrid-cloud autoscaling, workload prediction, and optimization techniques.

However, the reviewed literature indicates limited integration of future workload/resource prediction with coordinated scaling and workload-placement decisions across heterogeneous private and public cloud resources while simultaneously considering application performance, SLO requirements, resource availability, and operational cost.

## Preliminary Architecture

The proposed system follows a feedback-driven workflow:

```text
Telemetry
    ↓
Workload / Resource Prediction
    ↓
Decision & Optimization
    ↓
Scaling + Cloud Placement
    ↓
Performance & Cost Evaluation
    ↓
Feedback
    ↺
