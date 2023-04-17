// BTService_CheckPlayerInSight.h

#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/Services/BTService_BlackboardBase.h"
#include "BTService_CheckPlayerInSight.generated.h"

UCLASS()
class GAMEPLAYLIB_API UBTService_CheckPlayerInSight : public UBTService_BlackboardBase
{
    GENERATED_BODY()

public:
    UBTService_CheckPlayerInSight();

protected:
    virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;
};
