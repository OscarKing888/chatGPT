// BTTask_Roam.cpp

#include "BTTask_Roam.h"
#include "AIController.h"
#include "NavigationSystem.h"

UBTTask_Roam::UBTTask_Roam()
{
    NodeName = TEXT("Roam");
}

EBTNodeResult::Type UBTTask_Roam::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    AAIController* AIController = OwnerComp.GetAIOwner();
    if (AIController == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    APawn* ControlledPawn = AIController->GetPawn();
    if (ControlledPawn == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    FVector Origin = ControlledPawn->GetActorLocation();
    UNavigationSystemV1* NavSys = FNavigationSystem::GetCurrent<UNavigationSystemV1>(ControlledPawn->GetWorld());

    if (NavSys == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    FNavLocation RandomLocation;
    if (NavSys->GetRandomPointInNavigableRadius(Origin, 1000.0f, RandomLocation))
    {
        AIController->MoveToLocation(RandomLocation.Location);
        return EBTNodeResult::Succeeded;
    }

    return EBTNodeResult::Failed;
}
