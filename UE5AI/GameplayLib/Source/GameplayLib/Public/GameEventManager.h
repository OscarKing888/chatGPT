// EventManager.h
#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "GameEventManager.generated.h"

USTRUCT(BlueprintType)
struct FMessageData
{
    GENERATED_BODY()

    // ������ӱ�Ҫ���¼����ݳ�Ա
};


// ����һ��ͨ�õ��¼�����
DECLARE_DYNAMIC_DELEGATE_OneParam(FOnGameEvent, const FMessageData&, Params);

UCLASS()
class GAMEPLAYLIB_API UGameEventManager : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    // ʹ�þ�̬����ע���¼�
    UFUNCTION(BlueprintCallable, Category = "Event Manager", meta = (DisplayName = "Register Event"))
        static void RegisterEvent(const FName& EventID, const FOnGameEvent& Callback);

    // ʹ�þ�̬����ע���¼�
    UFUNCTION(BlueprintCallable, Category = "Event Manager")
        static void UnregisterEvent(const FName& EventID);

    // ʹ�þ�̬���������¼�
    UFUNCTION(BlueprintCallable, Category = "Event Manager")
        static void TriggerEvent(const FName& EventID, const FMessageData& Params);

private:
    // �洢�¼������ߵ�TMap
    static TMap<FName, FOnGameEvent> EventMap;
};
