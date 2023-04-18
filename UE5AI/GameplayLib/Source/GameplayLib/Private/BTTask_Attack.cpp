// BTTask_Attack.cpp

#include "BTTask_Attack.h"
#include "MonsterCharacter.h"
#include "AIController.h"
#include "BehaviorTree/BlackboardComponent.h"

UBTTask_Attack::UBTTask_Attack()
{
    NodeName = TEXT("Attack");
}

EBTNodeResult::Type UBTTask_Attack::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
    AAIController* AIController = OwnerComp.GetAIOwner();
    if (AIController == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    AMonsterCharacter* Monster = Cast<AMonsterCharacter>(AIController->GetPawn());
    if (Monster == nullptr)
    {
        return EBTNodeResult::Failed;
    }
    

    UBlackboardComponent* BlackboardComponent = AIController->GetBlackboardComponent();
    if (BlackboardComponent == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    AActor* AttackTarget = Cast<AActor>(BlackboardComponent->GetValueAsObject(GetSelectedBlackboardKey()));

    if (AttackTarget == nullptr)
    {
        return EBTNodeResult::Failed;
    }

    // Call a custom function in your MonsterCharacter to perform the attack
    Monster->Attack(AttackTarget);

    return EBTNodeResult::Succeeded;
}
